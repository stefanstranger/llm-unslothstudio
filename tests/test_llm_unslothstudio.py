from __future__ import annotations

import os
import subprocess
import sys

import httpx
import openai

import llm_unslothstudio


def test_cli_import_registers_sync_model_alias():
    script = """
import httpx


class ModelResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"id": "local-model"}]}


calls = 0


def get_models(*args, **kwargs):
    global calls
    calls += 1
    if calls == 1:
        raise httpx.ReadTimeout("Studio is still loading")
    return ModelResponse()


httpx.get = get_models

import llm.cli
import llm

assert llm.get_model("local-model").model_id == "unsloth:local-model"
assert llm.get_async_model("local-model").model_id == "unsloth:local-model"
assert calls == 2
"""
    environment = os.environ.copy()
    environment.pop("LLM_LOAD_PLUGINS", None)
    environment["UNSLOTH_STUDIO_API_KEY"] = "sk-unsloth-test"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_registers_discovered_model_with_aliases(monkeypatch):
    monkeypatch.setattr(
        llm_unslothstudio, "discover_model_ids", lambda: ["local-model"]
    )
    registered = []

    llm_unslothstudio.register_models(
        lambda *args, **kwargs: registered.append((args, kwargs))
    )

    (model, async_model), kwargs = registered[0]
    assert model.model_id == "unsloth:local-model"
    assert async_model.model_id == "unsloth:local-model"
    assert model.model_name == "local-model"
    assert kwargs["aliases"] == ("local-model", "unsloth")


def test_non_streaming_chat_completion(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    model = llm_unslothstudio.UnslothStudio(
        model_id="unsloth:local-model",
        model_name="local-model",
    )
    monkeypatch.setattr(
        model,
        "get_client",
        lambda key: openai.OpenAI(
            api_key=key,
            base_url=llm_unslothstudio.api_base_url(),
            http_client=httpx.Client(transport=transport),  # pyright: ignore
        ),
    )

    response = model.prompt("Say hello", key="sk-unsloth-test", stream=False)

    assert response.text() == "Hello!"
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer sk-unsloth-test"
    assert response.usage().input == 2
    assert response.usage().output == 1


def test_streaming_chat_completion(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'"stream":true' in request.content
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                '"created":1,"model":"local-model","choices":null,"usage":'
                '{"prompt_tokens":2,"completion_tokens":1,'
                '"total_tokens":3}}\n\n'
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                '"created":1,"model":"local-model","choices":[{"index":0,'
                '"delta":{"role":"assistant","content":"Hello"},'
                '"finish_reason":null}]}\n\n'
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                '"created":1,"model":"local-model","choices":[{"index":0,'
                '"delta":{"content":"!"},"finish_reason":null}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    model = llm_unslothstudio.UnslothStudio(
        model_id="unsloth:local-model",
        model_name="local-model",
    )
    monkeypatch.setattr(
        llm_unslothstudio.Chat,
        "get_client",
        lambda self, key: openai.OpenAI(
            api_key=key,
            base_url=llm_unslothstudio.api_base_url(),
            http_client=httpx.Client(  # pyright: ignore
                transport=httpx.MockTransport(handler)
            ),
        ),
    )

    response = model.prompt("Say hello", key="sk-unsloth-test")

    assert list(response) == ["Hello", "!"]
    assert response.text() == "Hello!"
    assert response.usage().input == 2
    assert response.usage().output == 1
