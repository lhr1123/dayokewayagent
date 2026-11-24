"""HTTP adapter and configuration tests with an in-process mock transport."""

import json
import os
from unittest.mock import patch

import httpx

from task1_agent.llm import ChatClient
from task1_agent.examples import demo_question
from task1_agent.graph import solve
from tests.support import GraphTestCase, checked_answer


class ModelClientTests(GraphTestCase):
    def test_environment_can_configure_limits_without_changing_model_defaults(self):
        settings = {"LLM_MODEL": "test-model", "LLM_BASE_URL": "https://example.com/v1", "LLM_API_KEY": "",
                    "LLM_READ_TIMEOUT_SECONDS": "45", "LLM_CONNECT_TIMEOUT_SECONDS": "5", "LLM_MAX_TOKENS": "800"}
        with patch.dict(os.environ, settings), ChatClient.from_env(demo=False) as model:
            self.assertEqual((model.http.timeout.read, model.http.timeout.connect, model.max_tokens), (45, 5, 800))
        with patch.dict(os.environ, {**settings, "LLM_READ_TIMEOUT_SECONDS": "nan"}):
            with self.assertRaisesRegex(ValueError, "LLM_READ_TIMEOUT_SECONDS"):
                ChatClient.from_env(demo=False)

    def test_demo_uses_json_completion_adapter(self):
        calls = []

        def handler(request):
            sent = json.loads(request.content)
            calls.append(sent)
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(sent["response_format"], {"type": "json_object"})
            payload = {"queries": [{"tool": "window", "start": 120}]} if len(calls) == 1 else checked_answer(
                "B", "altitude.change", 0, op="lt",
            )
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        with ChatClient("test-model", "https://example.com/v1", "", siliconflow=True) as model:
            model.http.close()
            model.http = httpx.Client(transport=httpx.MockTransport(handler))
            state = solve(self.graph, demo_question("segment")[0], model)
        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual(len(calls), 2)
        self.assertEqual([call["enable_thinking"] for call in calls], [False, False])
        self.assertEqual(state["model_calls"][1]["completion_tokens"], 5)

    def test_judge_repairs_invalid_schema_using_the_same_evidence(self):
        queries = {"queries": [{"tool": "window", "start": 120}]}
        good = checked_answer("B", "altitude.change", 0, op="lt")
        invalid = checked_answer("B", "altitude.change", 0, op="less_than")
        replies = iter((queries, invalid, good))
        payloads = []

        def handler(request):
            sent = json.loads(request.content)
            payloads.append(json.loads(sent["messages"][1]["content"]))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps(next(replies))}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            })

        with ChatClient("test-model", "https://example.com/v1", "") as model:
            model.http.close()
            model.http = httpx.Client(transport=httpx.MockTransport(handler))
            state = solve(self.graph, demo_question("segment")[0], model)

        self.assertEqual(state["result"]["结果"], "B")
        self.assertEqual(state["format_retries_used"], 1)
        self.assertEqual((state["revisions_used"], state["extra_rounds_used"]), (0, 0))
        self.assertEqual(payloads[1]["evidence"], payloads[2]["evidence"])
        self.assertEqual(payloads[2]["format_feedback"][0]["path"], "checks.0.op")
        self.assertNotIn("input", payloads[2]["format_feedback"][0])
        self.assertEqual(state["model_calls"][1]["completion_tokens"], 20)
        self.assertTrue(state["model_calls"][2]["format_retry"])

    def test_batch_enables_provider_options_for_explicit_siliconflow_endpoint(self):
        for host, siliconflow in (("api.siliconflow.cn", True), ("localhost:8000", False)):
            with self.subTest(host=host), patch.dict(os.environ, {
                "LLM_BASE_URL": f"https://{host}/v1", "LLM_MODEL": "test-model", "LLM_API_KEY": "",
            }), ChatClient.from_env(demo=False) as model:
                self.assertEqual(model.siliconflow, siliconflow)
