"""Model doubles and graph fixtures shared by workflow tests."""

from collections import deque
import unittest

from task1_agent.graph import build_graph


def checked_answer(answer, field, right, *, clause=1, evidence_id="E1", op="eq", region_id=None):
    left = {"id": evidence_id, "field": field, "region_id": region_id}
    return {"action": "answer", "answer": answer, "checks": [{"clause": clause, "left": left, "op": op, "right": right}]}


class ScriptedModel:
    """Capture graph requests and return validated synthetic responses."""

    def __init__(self, *responses):
        self.responses = deque(responses)
        self.payloads = []
        self.reasoning = []

    def ask(self, instructions, payload, schema, *, reasoning=False):
        self.payloads.append(payload)
        self.reasoning.append(reasoning)
        result = self.responses.popleft()
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result), {"model": "test"}


class GraphTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()
