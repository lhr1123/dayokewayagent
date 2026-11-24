"""Run one synthetic demo or write the results of a Task 1 batch."""

from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
from time import perf_counter
from typing import TextIO

from .graph import Task1State, build_graph, solve
from .llm import ChatClient
from .dataset import input_paths, read_questions
from .examples import OfflineDemoModel, demo_question
from .models import QuestionType


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _write(handle: TextIO, value: dict) -> None:
    handle.write(_json(value) + "\n")
    handle.flush()


def _trace(state: Task1State, elapsed_s: float) -> dict:
    return {
        "sample_id": state["sample_id"], "question_type": state["question_type"],
        "extra_rounds_used": state["extra_rounds_used"],
        "revisions_used": state["revisions_used"], "verification": state["verification"],
        "format_retries_used": state["format_retries_used"],
        "plan_format_retries_used": state["plan_format_retries_used"],
        "evidence": list(state["evidence"].values()), "decision": state.get("decision"),
        "model_calls": state["model_calls"], "elapsed_s": round(elapsed_s, 3),
        "result": state.get("result"), "error": state.get("error"),
    }


def run_demo(case: QuestionType, *, offline: bool = False) -> int:
    """Run a known-answer synthetic case with a scripted or hosted model."""
    graph = build_graph()
    question, expected = demo_question(case)
    client = nullcontext(OfflineDemoModel(case)) if offline else ChatClient.from_env(demo=True)
    with client as model:
        started = perf_counter()
        state = solve(graph, question, model)
    result = state.get("result", {})
    passed = result.get("结果") == expected
    print(_json({**_trace(state, perf_counter() - started), "expected": expected, "passed": passed}))
    return 0 if passed else 1


def run_batch(input_path: Path, output_path: Path, *, trace_path: Path | None = None,
              sample_ids: list[str] | None = None) -> int:
    """Process the requested samples and write JSONL artifacts."""
    sources = input_paths(input_path)
    questions = read_questions(sources)
    if sample_ids:
        ids = set(sample_ids)
        questions = [question for question in questions if question.sample_id in ids]
        if len(questions) != len(ids):
            raise ValueError("one or more requested sample IDs are absent")

    failures = output_path.with_suffix(".errors.jsonl")
    destinations = [output_path, failures, *([trace_path] if trace_path else [])]
    resolved = {path.resolve() for path in destinations}
    if len(resolved) != len(destinations) or resolved.intersection(path.resolve() for path in sources):
        raise ValueError("input, output, error and trace paths must be distinct")

    graph = build_graph()
    succeeded = 0
    with ExitStack() as stack:
        model = stack.enter_context(ChatClient.from_env(demo=False))
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        output = stack.enter_context(output_path.open("w", encoding="utf-8"))
        errors = stack.enter_context(failures.open("w", encoding="utf-8"))
        traces = stack.enter_context(trace_path.open("w", encoding="utf-8")) if trace_path else None
        for question in questions:
            started = perf_counter()
            state = solve(graph, question, model)
            if result := state.get("result"):
                _write(output, result)
                succeeded += 1
            else:
                _write(errors, {"sample_id": question.sample_id, **state["error"]})
            if traces is not None:
                _write(traces, _trace(state, perf_counter() - started))
    print(_json({"samples": len(questions), "answered": succeeded, "failed": len(questions) - succeeded}))
    return 0 if succeeded == len(questions) else 1
