"""LLM plugin for Unsloth Studio's chat and Decision APIs."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping
from functools import cache
from typing import Any

import httpx
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
DECISION_MODEL_ID = "unsloth:decision"
DECISION_TIMEOUT = 120.0
DISCOVERY_TIMEOUT = 10.0
KEY_ALIAS = "unslothstudio"
KEY_ENV_VAR = "UNSLOTH_STUDIO_API_KEY"
URL_ENV_VAR = "UNSLOTH_STUDIO_URL"


def api_base_url() -> str:
    """Return the configured OpenAI-compatible API base URL."""
    base_url = os.environ.get(URL_ENV_VAR, DEFAULT_BASE_URL).rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def resolve_key(explicit_key: str | None = None) -> str | None:
    """Resolve a key from input, LLM storage, or the environment."""
    return llm.get_key(input=explicit_key, alias=KEY_ALIAS, env=KEY_ENV_VAR)


@cache
def discover_model_ids() -> list[str]:
    """Return model IDs currently exposed by Unsloth Studio."""
    key = resolve_key()
    if not key:
        return []

    for attempt in range(2):
        try:
            response = httpx.get(
                f"{api_base_url()}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=DISCOVERY_TIMEOUT,
            )
            response.raise_for_status()
            return sorted(
                item["id"]
                for item in response.json().get("data", [])
                if isinstance(item, dict) and item.get("id")
            )
        except httpx.TimeoutException:
            if attempt == 0:
                continue
            return []
        except (httpx.HTTPError, TypeError, ValueError):
            return []

    return []


def _normalize_chunk(chunk: Any) -> Any:
    if chunk.choices is None:
        return chunk.model_copy(update={"choices": []})
    return chunk


def _normalize_chunks(chunks: Iterator[Any]) -> Iterator[Any]:
    for chunk in chunks:
        yield _normalize_chunk(chunk)


async def _normalize_async_chunks(
    chunks: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    async for chunk in chunks:
        yield _normalize_chunk(chunk)


class _CompletionsProxy:
    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, *args: Any, **kwargs: Any) -> Any:
        completion = self._completions.create(*args, **kwargs)
        if not kwargs.get("stream"):
            return completion
        if hasattr(completion, "__aiter__"):
            return _normalize_async_chunks(completion)
        return _normalize_chunks(completion)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _ChatProxy:
    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _ClientProxy:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _ChatProxy(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _UnslothStudioMixin:
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR

    def __init__(self, model_id: str, model_name: str) -> None:
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            api_base=api_base_url(),
            vision=True,
            supports_schema=True,
            supports_tools=True,
        )

    def __str__(self) -> str:
        return f"Unsloth Studio: {self.model_name}"

    def get_client(self, key: str | None = None) -> Any:
        return _ClientProxy(super().get_client(key))


class UnslothStudio(_UnslothStudioMixin, Chat):
    """Synchronous Unsloth Studio chat model."""


class AsyncUnslothStudio(_UnslothStudioMixin, AsyncChat):
    """Asynchronous Unsloth Studio chat model."""


class DecisionOptions(llm.Options):
    """Options accepted by the Unsloth Studio Decision API model."""

    questions: str | dict[str, Any] | None = None
    decision_model: str = "laya"


def _parse_json_object(value: str | Mapping[str, Any], name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as ex:
        raise ValueError(f"{name} must be valid JSON: {ex.msg}") from ex

    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _decision_payload(prompt: llm.Prompt) -> dict[str, Any]:
    options = prompt.options
    if not isinstance(options, DecisionOptions):
        raise TypeError("Decision API requires DecisionOptions")
    if options.questions is not None:
        questions = _parse_json_object(options.questions, "questions")
        if prompt.prompt is None:
            raise ValueError("Decision API state text is required")
        return {
            "model": options.decision_model,
            "state": prompt.prompt,
            "questions": questions,
        }

    if prompt.prompt is None:
        raise ValueError(
            "Provide a JSON request payload or state text with the questions option"
        )

    payload = _parse_json_object(prompt.prompt, "Decision API request")
    if "state" not in payload:
        raise ValueError("Decision API request must include state")
    if not isinstance(payload.get("questions"), dict):
        raise ValueError("Decision API request questions must be a JSON object")
    payload.setdefault("model", options.decision_model)
    return payload


def _decision_headers(key: str | None) -> dict[str, str]:
    if not key:
        return {}
    return {"Authorization": "Bearer " + key}


def _decision_response_text(
    api_response: httpx.Response, response: llm.Response | llm.AsyncResponse
) -> str:
    api_response.raise_for_status()
    data = api_response.json()
    if not isinstance(data, dict):
        raise ValueError("Decision API response must be a JSON object")

    usage = data.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        response.set_usage(
            input=input_tokens if isinstance(input_tokens, int) else None,
            output=output_tokens if isinstance(output_tokens, int) else None,
        )

    return json.dumps(data, ensure_ascii=False)


class _UnslothDecisionMixin:
    model_id = DECISION_MODEL_ID
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR
    can_stream = False
    Options = DecisionOptions

    def __str__(self) -> str:
        return "Unsloth Studio: Decision API"


class UnslothDecision(_UnslothDecisionMixin, llm.KeyModel):
    """Synchronous Unsloth Studio Decision API model."""

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None,
        key: str | None,
    ) -> Iterator[str]:
        del stream, conversation
        api_response = httpx.post(
            f"{api_base_url()}/systemone",
            headers=_decision_headers(key),
            json=_decision_payload(prompt),
            timeout=DECISION_TIMEOUT,
        )
        yield _decision_response_text(api_response, response)


class AsyncUnslothDecision(_UnslothDecisionMixin, llm.AsyncKeyModel):
    """Asynchronous Unsloth Studio Decision API model."""

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None,
        key: str | None,
    ) -> AsyncGenerator[str, None]:
        del stream, conversation
        async with httpx.AsyncClient() as client:
            api_response = await client.post(
                f"{api_base_url()}/systemone",
                headers=_decision_headers(key),
                json=_decision_payload(prompt),
                timeout=DECISION_TIMEOUT,
            )
        yield _decision_response_text(api_response, response)


@llm.hookimpl
def register_models(register) -> None:
    """Register Unsloth Studio chat models and its Decision API."""
    model_ids = discover_model_ids()
    for model_name in model_ids:
        aliases = [model_name]
        if len(model_ids) == 1:
            aliases.append("unsloth")
        model_id = f"unsloth:{model_name}"
        register(
            UnslothStudio(model_id, model_name),
            AsyncUnslothStudio(model_id, model_name),
            aliases=tuple(aliases),
        )

    register(
        UnslothDecision(),
        AsyncUnslothDecision(),
        aliases=("unsloth-decision",),
    )
