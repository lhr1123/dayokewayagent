"""Evidence validation and revision tests."""

from task1_agent.examples import demo_question
from task1_agent.graph import solve
from tests.support import GraphTestCase, ScriptedModel, checked_answer


class VerificationTests(GraphTestCase):
    def test_explicit_approximation_allows_rounding_but_preserves_sign(self):
        question, _ = demo_question("segment")
        question.instruction = "哪一项正确？\nA. 末段以约2.8米/秒下降。"
        question.track.points[-1].altitude = 1431.953
        plan = {"queries": [{"tool": "window", "start": 120}]}
        wrong = checked_answer("A", "altitude.mean_rate_per_s", 2.8)
        correct = checked_answer("A", "altitude.mean_rate_per_s", -2.8)
        state = solve(self.graph, question, ScriptedModel(plan, wrong, correct))

        self.assertEqual(state["result"]["结果"], "A")
        self.assertEqual([attempt["passed"] for attempt in state["verification"]], [False, True])

    def test_region_limit_shorthand_needs_an_unambiguous_visit(self):
        question, _ = demo_question("region")
        question.instruction = "哪一项正确？\nA. 区域 A 内有超速记录。"
        plan = {"queries": [{"tool": "regions", "limits": [{"metric": "speed", "maximum": 20}]}]}
        answer = checked_answer("A", "regions.A.limits[0].observations_outside", 0, op="gt")
        ambiguous = solve(self.graph, question, ScriptedModel(plan, answer, answer))
        self.assertIn("ambiguous", ambiguous["error"]["message"])

        question.track.points[0].lon = 1.5
        single = solve(self.graph, question, ScriptedModel(plan, answer))
        self.assertEqual(single["result"]["结果"], "A")

    def test_conflicting_number_is_returned_for_a_single_revision(self):
        query = {"tool": "window", "start": 120, "end": 180, "metrics": ["altitude"]}
        bad = checked_answer("A", "altitude.change", 0, op="gt")
        good = checked_answer("B", "altitude.last", {"id": "E1", "field": "altitude.first"}, op="lt")
        question, _ = demo_question("segment")
        model = ScriptedModel({"queries": [query]}, bad, good)
        state = solve(self.graph, question, model)

        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual(state["revisions_used"], 1)
        self.assertEqual(state["extra_rounds_used"], 0)
        self.assertEqual([check["passed"] for check in state["verification"]], [False, True])
        feedback = model.payloads[-1]["verification_feedback"]["issues"][0]
        self.assertEqual((feedback["left_actual"], feedback["right_actual"]), (-300, 0))

    def test_region_coverage_rejects_whole_track_window_for_inside_claim(self):
        question, _ = demo_question("region")
        question.instruction = "哪一项正确？\nA. 区域 A 内最低高度为 40。\nB. 区域 A 内最低高度为 100。"
        window = {"tool": "window", "metrics": ["altitude"]}
        bad = checked_answer("A", "altitude.min", 40, evidence_id="E2")
        good = checked_answer("B", "regions.A.visits.1.motion.altitude.min", 100)
        model = ScriptedModel({"queries": [window]}, bad, good)
        state = solve(self.graph, question, model)

        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual([e["query"]["tool"] for e in model.payloads[1]["evidence"]], ["regions", "window"])
        self.assertIn("not inside region A", state["verification"][0]["issues"][0]["message"])

    def test_missing_clause_stops_after_one_revision(self):
        question, _ = demo_question("segment")
        question.instruction = "哪一项正确？\nA. 末段下降；最高高度为 1500 米。"
        query = {"tool": "window", "start": 120, "metrics": ["altitude"]}
        incomplete = checked_answer("A", "altitude.change", 0, op="lt")
        state = solve(self.graph, question, ScriptedModel({"queries": [query]}, incomplete, incomplete))

        self.assertNotIn("result", state)
        self.assertEqual(state["error"]["node"], "verify")
        self.assertEqual(state["revisions_used"], 1)
        self.assertEqual(state["verification"][-1]["issues"][0]["quote"], "最高高度为 1500 米")

    def test_literal_zero_cannot_be_replaced_by_a_convenient_bound(self):
        question, _ = demo_question("segment")
        question.instruction = "哪一项正确？\nA. 高度与速度均为 0。"
        for point in question.track.points:
            point.altitude, point.speed = 0, 1
        bad = checked_answer("A", "speed.max", 2, op="le")
        bad["checks"].extend(checked_answer("A", "altitude.max", 0)["checks"])
        state = solve(self.graph, question, ScriptedModel({"queries": [{"tool": "window"}]}, bad, bad))

        self.assertNotIn("result", state)
        self.assertEqual(state["verification"][0]["issues"][0]["observed_max"], 1)
        self.assertEqual(state["verification"][0]["issues"][0]["required_value"], 0)

    def test_exact_window_conflict_removes_option_from_revision(self):
        question, _ = demo_question("segment")
        question.instruction = "哪一项正确？\nA. 00时01分40秒至00时02分40秒速度始终为0。\nB. 末段速度大于0。"
        window = {"tool": "window", "start": 120, "end": 180, "metrics": ["speed"]}
        bad = checked_answer("A", "speed.max", 100, op="le")
        good = checked_answer("B", "speed.min", 0, op="gt")
        model = ScriptedModel({"queries": [window]}, bad, good)
        state = solve(self.graph, question, model)

        self.assertEqual(list(model.payloads[-1]["options"]), ["B"])
        self.assertTrue(state["verification"][0]["issues"][0]["excludes_option"])
        self.assertEqual(state["result"]["结果"], "B")

    def test_region_phase_clause_cannot_use_whole_visit_as_phase_duration(self):
        question, _ = demo_question("region")
        question.instruction = "哪一项正确？\nA. 区域 A 内起飞阶段持续时间长于爬升阶段。"
        for point in question.track.points:
            point.lon = 0.5
        misleading = checked_answer("A", "regions.A.visits.0.duration_s", 0, op="gt")
        model = ScriptedModel({"queries": [{"tool": "regions", "region_ids": ["A"]}]},
                              misleading, {"action": "insufficient_evidence", "reason": "no phase boundary"})
        state = solve(self.graph, question, model)

        self.assertNotIn("result", state)
        visit = model.payloads[1]["evidence"][0]["output"]["regions"]["A"]["visits"][0]
        self.assertEqual(len(visit["motion_bins"]), 4)
        self.assertIn("distinct local subwindows", state["verification"][0]["issues"][0]["message"])
