"""Environment-based settings for the model client."""

from dataclasses import dataclass, field
from math import isfinite
import os
from urllib.parse import urlsplit


DEFAULT_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_TOKENS = 2400
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_HOST = "api.siliconflow.cn"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


@dataclass(frozen=True)
class ModelSettings:
    """Resolve target, credentials, and request limits before starting a run."""

    model: str
    base_url: str
    api_key: str = field(repr=False)
    read_timeout_s: float = DEFAULT_READ_TIMEOUT_SECONDS
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS

    @property
    def siliconflow(self) -> bool:
        return urlsplit(self.base_url).hostname == SILICONFLOW_HOST

    @classmethod
    def from_env(cls, *, demo: bool) -> "ModelSettings":
        if demo:
            key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
            if not key:
                raise ValueError("SILICONFLOW_API_KEY is required for the online demo")
            model = os.environ.get("SILICONFLOW_MODEL", SILICONFLOW_DEFAULT_MODEL)
            base_url = SILICONFLOW_BASE_URL
        else:
            model = os.environ.get("LLM_MODEL", "").strip()
            base_url = os.environ.get("LLM_BASE_URL", "").strip()
            key = os.environ.get("LLM_API_KEY", "")
            if not model or not base_url:
                raise ValueError("set LLM_MODEL and LLM_BASE_URL to run on dataset files")

        return cls(
            model, base_url, key,
            read_timeout_s=_timeout("LLM_READ_TIMEOUT_SECONDS", DEFAULT_READ_TIMEOUT_SECONDS),
            connect_timeout_s=_timeout("LLM_CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT_SECONDS),
            max_tokens=_max_tokens(),
        )


def _timeout(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _max_tokens() -> int:
    try:
        value = int(os.environ.get("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    except ValueError as exc:
        raise ValueError("LLM_MAX_TOKENS must be a positive integer") from exc
    if value <= 0:
        raise ValueError("LLM_MAX_TOKENS must be a positive integer")
    return value
