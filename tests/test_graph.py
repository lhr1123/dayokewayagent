"""Graph routing, supplemental evidence, and retry-budget tests."""

from task1_agent.llm import ModelCallError, ModelResponseError
from task1_agent.examples import demo_question
from task1_agent.graph import solve
from tests.support import GraphTestCase, ScriptedModel, checked_answer


class GraphTests(GraphTestCase):
    def test_segment_path_and_equivalent_window_deduplication(self):
        query = {"tool": "window", "start": 120, "end": 180, "metrics": ["altitude"]}
        utc = {**query, "start": "2025-01-02T00:01:40Z", "end": "2025-01-02T00:02:40Z"}
        model = ScriptedModel({"queries": [query, utc]},
                              checked_answer("B", "altitude.change", 0, op="lt"))
        question, expected = demo_question("segment")

        state = solve(self.graph, question, model)

        self.assertEqual(state["result"], {"样本编号": "demo_segment", "结果": expected})
        self.assertEqual(len(model.payloads), 2)
        self.assertEqual(model.payloads[0]["track"]["available_tools"], ["window"])
        self.assertEqual(len(model.payloads[1]["evidence"]), 1)
        summary = model.payloads[1]["evidence"][0]["output"]["altitude"]
        self.assertEqual((summary["first"], summary["last"], summary["mean_rate_per_s"]), (1600, 1300, -5))

    def test_region_path_adds_supplement_without_losing_existing_evidence(self):
        first = {"tool": "regions", "region_ids": ["A"]}
        more = {"tool": "window", "start": 25, "end": 35, "metrics": ["altitude"]}
        model = ScriptedModel({"queries": [first]}, {"action": "fetch_more", "queries": [more]},
                              checked_answer("C", "altitude.change", 0, evidence_id="E2", op="gt"))
        question, expected = demo_question("region")

        state = solve(self.graph, question, model)

        self.assertEqual(state["result"]["结果"], expected)
        self.assertEqual(state["extra_rounds_used"], 1)
        self.assertEqual(model.payloads[0]["track"]["region_ids"], ["A"])
        first, second = model.payloads[-1]["evidence"]
        region = first["output"]["regions"]["A"]
        self.assertEqual((region["entry_count"], region["total_duration_s"]), (1, 15))
        self.assertTrue(region["inside_at_window_start"])
        self.assertIsNone(region["visits"][0]["entry"])
        self.assertEqual(region["visits"][1]["exit"]["utc"], "2025-01-02T00:00:05Z")
        self.assertEqual(second["output"]["altitude"]["change"], 50)
        self.assertFalse(model.payloads[-1]["can_fetch"])
        self.assertEqual(model.reasoning, [False, True, False])

    def test_request_after_one_supplement_ends_with_error(self):
        model = ScriptedModel(
            {"queries": [{"tool": "window", "start": 0, "end": 60}]},
            {"action": "fetch_more", "queries": [{"tool": "window", "start": 60, "end": 120}]},
            {"action": "fetch_more", "queries": [{"tool": "window", "start": 120, "end": 180}]},
        )
        state = solve(self.graph, demo_question("segment")[0], model)
        self.assertEqual(len(model.payloads), 3)
        self.assertEqual(state["error"]["node"], "judge")
        self.assertNotIn("result", state)

    def test_plan_error_ends_without_an_answer(self):
        state = solve(self.graph, demo_question("segment")[0], ScriptedModel(ModelCallError("timeout")))
        self.assertEqual(state["error"], {"node": "plan", "message": "timeout"})
        self.assertNotIn("result", state)

    def test_plan_retries_out_of_range_bins_with_field_feedback(self):
        issue = ModelResponseError("Plan", [{"path": "queries.0.window.bins", "type": "less_than_equal",
                                             "message": "Input should be less than or equal to 12"}], {})
        answer = checked_answer("B", "altitude.change", 0, op="lt")
        model = ScriptedModel(issue, {"queries": [{"tool": "window", "start": 120, "bins": 2}]}, answer)

        state = solve(self.graph, demo_question("segment")[0], model)

        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual((state["plan_format_retries_used"], state["format_retries_used"]), (1, 0))
        self.assertEqual(model.payloads[1]["format_feedback"][0]["path"], "queries.0.window.bins")

    def test_repeat_only_supplement_stops_without_requerying(self):
        query = {"tool": "window", "start": 0, "end": 60}
        model = ScriptedModel({"queries": [query]}, {"action": "fetch_more", "queries": [query]})
        state = solve(self.graph, demo_question("segment")[0], model)
        self.assertEqual(len(model.payloads), 2)
        self.assertEqual(state["extra_rounds_used"], 0)
        self.assertIn("no new valid evidence", state["error"]["message"])

    def test_judge_stops_after_the_format_repair_budget(self):
        bad = ModelResponseError("Decision", [{"path": "action", "type": "missing", "message": "Field required"}], {})
        model = ScriptedModel({"queries": [{"tool": "window"}]}, bad, bad)

        state = solve(self.graph, demo_question("segment")[0], model)

        self.assertNotIn("result", state)
        self.assertEqual(state["format_retries_used"], 1)
        self.assertEqual(state["verification"], [])
        self.assertEqual(len(model.payloads), 3)
        self.assertIn("at action", state["error"]["message"])

    def test_judge_transport_error_does_not_consume_a_format_retry(self):
        model = ScriptedModel({"queries": [{"tool": "window"}]}, ModelCallError("model request failed: ReadTimeout"))
        state = solve(self.graph, demo_question("segment")[0], model)

        self.assertEqual(state["format_retries_used"], 0)
        self.assertNotIn("result", state)
        self.assertEqual(len(model.payloads), 2)

    def test_judge_repairs_format_on_a_verification_revision(self):
        wrong = checked_answer("A", "altitude.change", 0, op="gt")
        corrected = checked_answer("B", "altitude.change", 0, op="lt")
        bad_format = ModelResponseError("Decision", [
            {"path": "reason", "type": "string_too_long", "message": "Reason is too long"},
        ], {})
        model = ScriptedModel({"queries": [{"tool": "window", "start": 120}]}, wrong, bad_format, corrected)

        state = solve(self.graph, demo_question("segment")[0], model)

        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual((state["revisions_used"], state["format_retries_used"]), (1, 1))
        self.assertEqual([item["passed"] for item in state["verification"]], [False, True])
        self.assertEqual(model.payloads[-1]["verification_feedback"]["answer"], "A")
        self.assertEqual(model.payloads[-1]["format_feedback"][0]["path"], "reason")

    def test_revision_can_use_remaining_supplement(self):
        question, _ = demo_question("segment")
        first = {"tool": "window", "start": 0, "end": 60, "metrics": ["altitude"]}
        later = {**first, "start": 120, "end": 180}
        wrong = checked_answer("B", "altitude.change", 0, op="lt")
        corrected = checked_answer("B", "altitude.change", 0, op="lt", evidence_id="E2")
        model = ScriptedModel({"queries": [first]}, wrong, {"action": "fetch_more", "queries": [later]}, corrected)
        state = solve(self.graph, question, model)

        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual((state["revisions_used"], state["extra_rounds_used"]), (1, 1))
        self.assertEqual([item["passed"] for item in state["verification"]], [False, True])
