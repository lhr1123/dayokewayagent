"""Numerical evidence and geometry tests."""

import unittest

from task1_agent.evidence import EvidenceCalculator
from task1_agent.examples import demo_question
from task1_agent.models import WindowQuery


class EvidenceTests(unittest.TestCase):
    def test_missing_measurements_and_inclusive_limits(self):
        question, _ = demo_question("segment")
        question.track.points[1].altitude = None
        result = EvidenceCalculator(question).execute(WindowQuery.model_validate({
            "tool": "window", "start": 0, "end": 60, "metrics": ["altitude"],
            "limits": [{"metric": "altitude", "maximum": 1600}],
        }))
        self.assertEqual(result["altitude"]["missing"], 1)
        self.assertEqual(result["limits"][0]["observations_outside"], 0)
        self.assertEqual(result["limits"][0]["interpolated_coverage_s"], 0)

    def test_window_can_be_anchored_to_last_region_exit(self):
        question, _ = demo_question("region")
        query = WindowQuery.model_validate({
            "tool": "window", "start": 0, "end": 10, "metrics": ["altitude"],
            "anchor": {"region_id": "A", "event": "exit", "occurrence": "last"},
        })
        result = EvidenceCalculator(question).execute(query)
        self.assertEqual((result["start"]["s"], result["end"]["s"]), (25, 35))
        self.assertEqual(result["anchor_at"]["s"], 25)
        self.assertEqual(result["altitude"]["change"], 50)
