"""One-question LangGraph run with one optional evidence round."""

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import TypeAdapter

from .evidence import EvidenceCalculator
from .llm import ModelClient, ask_with_format_repair
from .models import Answer, Decision, Plan, Query, Question, QuestionType, RegionQuery
from .prompts import JUDGE_PROMPT, PLAN_PROMPT, REGION_PLAN_GUIDE
from .verification import option_checks, split_options


class Task1State(TypedDict, total=False):
    """Serializable sample data, working evidence, and final status."""

    sample_id: str
    question_type: QuestionType
    question: str
    track_ref: str
    context_summary: dict
    pending_queries: list[dict]
    evidence: dict[str, dict]
    extra_rounds_used: int
    revisions_used: int
    format_retries_used: int
    plan_format_retries_used: int
    verification: list[dict]
    decision: dict
    answer: Answer
    result: dict
    error: dict[str, str]
    model_calls: list[dict]


@dataclass(frozen=True)
class Task1Context:
    """Per-run dependencies injected outside the persisted state."""

    model: ModelClient
    calculator: EvidenceCalculator


QUERY_ADAPTER = TypeAdapter(Query)


def prepare(state: Task1State, runtime: Runtime[Task1Context]) -> dict:
    """Summarize common coverage and initialize per-sample accumulation."""
    return {"context_summary": runtime.context.calculator.summary(), "pending_queries": [],
            "evidence": {}, "extra_rounds_used": 0, "revisions_used": 0,
            "format_retries_used": 0, "plan_format_retries_used": 0,
            "verification": [], "model_calls": []}


def segment_context(state: Task1State) -> dict:
    """Expose the window tool for segmented motion questions."""
    return {"context_summary": {**state["context_summary"], "available_tools": ["window"]}}


def region_context(state: Task1State, runtime: Runtime[Task1Context]) -> dict:
    """Expose the region IDs and both evidence tools for regional questions."""
    return {"context_summary": {**state["context_summary"], "available_tools": ["window", "regions"],
                                "region_ids": list(runtime.context.calculator.regions)}}


def plan(state: Task1State, runtime: Runtime[Task1Context]) -> dict:
    """Ask the model to propose a batch of evidence requests."""
    instructions = PLAN_PROMPT + REGION_PLAN_GUIDE if state["question_type"] == "region" else PLAN_PROMPT
    result, calls, retried, error = ask_with_format_repair(
        runtime.context.model, "plan", instructions,
        {"question": state["question"], "track": state["context_summary"]}, Plan,
    )
    update = {"model_calls": calls, "plan_format_retries_used": int(retried)}
    if result is None:
        return {**update, "error": error}
    queries = result.queries
    region_ids = set(state["context_summary"].get("region_ids", []))
    if state["question_type"] == "region" and not any(
        isinstance(query, RegionQuery) and (query.region_ids is None or
                                           bool(query.region_ids) and set(query.region_ids) <= region_ids)
        for query in queries
    ):
        queries = [RegionQuery(tool="regions"), *queries[:15]]
    return {**update, "pending_queries": [query.model_dump(mode="json") for query in queries]}


def _query_key(calculator: EvidenceCalculator, query: Query) -> str:
    try:
        return calculator.query_key(query)
    except ValueError:
        return query.model_dump_json()


def collect(state: Task1State, runtime: Runtime[Task1Context]) -> dict:
    """Calculate and cache all evidence in the current batch."""
    calculator = runtime.context.calculator
    evidence = dict(state["evidence"])
    for request in state["pending_queries"]:
        query = QUERY_ADAPTER.validate_python(request)
        key = _query_key(calculator, query)
        if key in evidence:
            continue
        try:
            if query.tool not in state["context_summary"]["available_tools"]:
                raise ValueError("tool is unavailable for this question type")
            output = calculator.execute(query)
        except ValueError as exc:
            output = {"unavailable": str(exc)}
        evidence[key] = {"id": f"E{len(evidence) + 1}", "query": request, "output": output}
    return {"pending_queries": [], "evidence": evidence}


def judge(state: Task1State, runtime: Runtime[Task1Context]) -> dict:
    """Decide an answer or accept one batch of genuinely new queries."""
    can_fetch = state["extra_rounds_used"] == 0
    evidence = list(state["evidence"].values())
    has_window = any(item["query"]["tool"] == "window" and item["output"].get("observations", 0) > 0
                     for item in evidence)
    has_region_detail = any(
        sum(b["altitude"]["observations"] > 0 or b["speed"]["observations"] > 0 for b in visit.get("motion_bins", [])) >= 2
        for item in evidence for region in item["output"].get("regions", {}).values() for visit in region["visits"]
    )
    has_detail = has_window if state["question_type"] == "segment" else has_region_detail
    stem, options = split_options(state["question"])
    excluded = _excluded_answers(state["verification"])
    options = {letter: dict(enumerate(clauses, start=1)) for letter, clauses in options.items() if letter not in excluded}
    payload = {"question": stem, "options": options, "track": state["context_summary"],
               "evidence": evidence, "can_fetch": can_fetch}
    if state["verification"]:
        payload["verification_feedback"] = state["verification"][-1]
    reasoning = can_fetch and not has_detail and not state["revisions_used"]
    decision, calls, retried, error = ask_with_format_repair(
        runtime.context.model, "judge", JUDGE_PROMPT, payload, Decision,
        allow_repair=state["format_retries_used"] == 0, reasoning=reasoning,
    )
    update = {"model_calls": [*state["model_calls"], *calls],
              "format_retries_used": state["format_retries_used"] + int(retried)}
    if decision is None:
        return {**update, "error": error}
    update["decision"] = decision.model_dump(mode="json")
    if decision.answer is not None:
        return update
    if decision.action == "insufficient_evidence":
        return {**update, "error": {"node": "judge", "message": "insufficient evidence: " + decision.reason}}
    if can_fetch:
        calculator = runtime.context.calculator
        requests = {}
        for query in decision.queries:
            try:
                key = calculator.query_key(query)
            except ValueError:
                continue
            if query.tool not in state["context_summary"]["available_tools"]:
                continue
            if key not in state["evidence"]:
                requests[key] = query.model_dump(mode="json")
        if requests:
            return {**update, "extra_rounds_used": 1, "pending_queries": list(requests.values())}
        return {**update, "error": {"node": "judge", "message": "no new valid evidence query was supplied"}}
    return {**update, "error": {"node": "judge", "message": "model requested evidence after the supplement limit"}}


def verify(state: Task1State) -> dict:
    """Accept checked answers or send precise feedback for one revision."""
    decision = Decision.model_validate(state["decision"])
    _, options = split_options(state["question"])
    issues = option_checks(decision, options, list(state["evidence"].values()),
                           state["context_summary"].get("region_ids", []))
    if decision.answer in _excluded_answers(state["verification"]):
        issues.insert(0, {"message": "That option was excluded by a previously verified conflict."})
    attempt = {"answer": decision.answer, "checks": state["decision"]["checks"],
               "passed": not issues, "issues": issues}
    update = {"verification": [*state["verification"], attempt]}
    if not issues:
        return {**update, "answer": decision.answer}
    if state["revisions_used"] == 0:
        return {**update, "revisions_used": 1}
    return {**update, "error": {"node": "verify", "message": issues[0]["message"]}}


def _excluded_answers(history: list[dict]) -> set[str]:
    return {attempt["answer"] for attempt in history if any(i.get("excludes_option") for i in attempt["issues"])}


def finish(state: Task1State) -> dict:
    """Emit exactly the Task 1 answer fields."""
    return {"result": {"样本编号": state["sample_id"], "结果": state["answer"]}}


def route_question(state: Task1State) -> Literal["segment_context", "region_context"]:
    """Choose the context builder prescribed by the source file."""
    return "segment_context" if state["question_type"] == "segment" else "region_context"


def route_plan(state: Task1State) -> Literal["collect", "__end__"]:
    """Stop if the model could not produce the initial plan."""
    return END if state.get("error") else "collect"


def route_judge(state: Task1State) -> Literal["verify", "collect", "__end__"]:
    """Route using validated state, including the one-round limit."""
    if state.get("error"):
        return END
    if state["decision"]["action"] == "answer":
        return "verify"
    return "collect"


def route_verify(state: Task1State) -> Literal["finish", "judge", "__end__"]:
    if state.get("error"):
        return END
    return "finish" if state.get("answer") else "judge"


def build_graph():
    """Compile the graph once for reuse across samples."""
    graph = StateGraph(Task1State, context_schema=Task1Context)
    for node in (prepare, segment_context, region_context, plan, collect, judge, verify, finish):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", route_question)
    graph.add_edge("segment_context", "plan")
    graph.add_edge("region_context", "plan")
    graph.add_conditional_edges("plan", route_plan)
    graph.add_edge("collect", "judge")
    graph.add_conditional_edges("judge", route_judge)
    graph.add_conditional_edges("verify", route_verify)
    graph.add_edge("finish", END)
    return graph.compile()


def solve(graph, question: Question, model: ModelClient) -> Task1State:
    """Run a new sample with independent state and tool access."""
    return graph.invoke({"sample_id": question.sample_id, "question_type": question.kind,
                         "question": question.instruction, "track_ref": question.track.track_id},
                        context=Task1Context(model=model, calculator=EvidenceCalculator(question)))
