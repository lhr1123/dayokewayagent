"""JSON model adapter over an OpenAI-compatible completion endpoint."""

import json
from time import perf_counter
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_MAX_TOKENS, DEFAULT_READ_TIMEOUT_SECONDS, ModelSettings
from .prompts import FORMAT_REPAIR_PROMPT

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ModelCallError(RuntimeError):
    """A request failed or did not return the expected object."""


class ModelResponseError(ModelCallError):
    """A completion arrived but needs a corrected output object."""

    def __init__(self, schema_name: str, issues: list[dict[str, str]], usage: dict) -> None:
        first = issues[0]
        super().__init__(f"invalid model response for {schema_name} at {first['path']}: {first['message']}")
        self.issues = issues
        self.usage = usage


class ModelClient(Protocol):
    """Model dependency exposed to the graph."""

    def ask(self, instructions: str, payload: dict, schema: type[ResponseModel], *, reasoning: bool = False) -> tuple[ResponseModel, dict]:
        """Return validated model data and request metadata."""
        ...


class ChatClient:
    """Reusable HTTP client for a single configured model service."""

    def __init__(self, model: str, base_url: str, key: str, *, siliconflow: bool = False,
                 read_timeout_s: float = DEFAULT_READ_TIMEOUT_SECONDS,
                 connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.siliconflow = siliconflow
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(headers=headers, timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s))

    @classmethod
    def from_env(cls, *, demo: bool) -> "ChatClient":
        """Select the synthetic demo backend or an explicitly configured run backend."""
        settings = ModelSettings.from_env(demo=demo)
        return cls(settings.model, settings.base_url, settings.api_key, siliconflow=settings.siliconflow,
                   read_timeout_s=settings.read_timeout_s, connect_timeout_s=settings.connect_timeout_s,
                   max_tokens=settings.max_tokens)

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.http.close()

    def ask(self, instructions: str, payload: dict, schema: type[ResponseModel], *, reasoning: bool = False) -> tuple[ResponseModel, dict]:
        """Ask for a JSON object, validate it, and record safe timing and usage."""
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.siliconflow:
            request.update(response_format={"type": "json_object"}, enable_thinking=reasoning)
        started = perf_counter()
        try:
            response = self.http.post(self.url, json=request)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelCallError(f"model endpoint returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise ModelCallError(f"model request failed: {type(exc).__name__}") from exc

        try:
            body = response.json()
            choice = body["choices"][0]
            if not isinstance(choice, dict):
                raise ValueError("choice is not an object")
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise ModelCallError("invalid completion envelope") from exc

        reported_usage = body.get("usage")
        usage = reported_usage if isinstance(reported_usage, dict) else {}
        metadata = {
            "model": self.model, "elapsed_s": round(perf_counter() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        }
        if choice.get("finish_reason") == "length":
            raise ModelResponseError(schema.__name__, [{
                "path": "$", "type": "output_truncated", "message": "Output was truncated. Return a shorter complete JSON object.",
            }], metadata)
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError(schema.__name__, [{
                "path": "$", "type": "empty_content", "message": "Return a non-empty JSON object in the response content.",
            }], metadata)
        content = content.strip()
        if content.startswith("```") and content.endswith("```") and "\n" in content:
            content = content.partition("\n")[2][:-3].strip()
        try:
            parsed = schema.model_validate_json(content)
        except ValidationError as exc:
            issues = []
            for error in exc.errors(include_input=False, include_context=False, include_url=False)[:8]:
                path = ".".join(str(part) for part in error["loc"])[:160] or "$"
                issues.append({"path": path, "type": error["type"], "message": error["msg"][:180]})
            raise ModelResponseError(schema.__name__, issues, metadata) from exc
        return parsed, metadata


def ask_with_format_repair(model: ModelClient, node: str, instructions: str, payload: dict, schema: type[ResponseModel],
                     *, allow_repair: bool = True, reasoning: bool = False
                     ) -> tuple[ResponseModel | None, list[dict], bool, dict[str, str] | None]:
    """Regenerate at most once with field feedback and preserve diagnostics for both attempts."""
    calls = []
    for retry in (False, True) if allow_repair else (False,):
        try:
            result, usage = model.ask(instructions, payload, schema, reasoning=reasoning)
        except ModelResponseError as exc:
            calls.append({"node": node, **exc.usage, "format_retry": retry, "format_issues": exc.issues})
            error = {"node": node, "message": str(exc)}
            if allow_repair and not retry:
                payload = {**payload, "format_feedback": exc.issues}
                instructions += FORMAT_REPAIR_PROMPT
                continue
            return None, calls, retry, error
        except ModelCallError as exc:
            if retry:
                calls.append({"node": node, "format_retry": True, "error": str(exc)})
            return None, calls, retry, {"node": node, "message": str(exc)}
        calls.append({"node": node, **usage, "format_retry": retry})
        return result, calls, retry, None
