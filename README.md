---
title: LLM Unsloth Studio
description: Use models served by Unsloth Studio with the LLM command-line tool
---
## Installation

Install this plugin in the same environment as
[LLM](https://llm.datasette.io/):

```bash
llm install llm-unslothstudio
```

For local development, install this checkout as an editable plugin:

```bash
llm install -e .
```

## Configuration

Start Unsloth Studio with a model loaded. `unsloth run` is an alias for
`unsloth studio run` and uses `http://127.0.0.1:8888` by default:

```bash
unsloth run --model "unsloth/Qwen3.8-27B-GGUF"
```

If the GGUF repository contains multiple quantizations, select one explicitly:

```bash
unsloth run --model "unsloth/Qwen3.8-27B-GGUF" --gguf-variant "UD-Q4_K_XL"
```

Running `unsloth studio` without the `run` command starts the Studio service
but does not necessarily load a model. If Studio is already running, load the
model in the UI or enable **Settings > API > Model auto-switch**. Model
auto-switch lets Studio load the model requested by an LLM command.

Create an API key in **Settings > API**, then store it with LLM:

```bash
llm keys set unslothstudio
```

Paste the `sk-unsloth-...` key when prompted. You can use the
`UNSLOTH_STUDIO_API_KEY` environment variable instead.

Set `UNSLOTH_STUDIO_URL` when Studio uses another port or host. Supply the
server root or its `/v1` URL, and use the same port when starting Studio:

```powershell
unsloth run --model "unsloth/Qwen3.8-27B-GGUF" --port 8000
$env:UNSLOTH_STUDIO_URL = "http://127.0.0.1:8000"
```

## Usage

The plugin discovers models from `GET /v1/models` when LLM starts. List them:

```bash
llm models list
```

Each model has a canonical ID prefixed with `unsloth:` that avoids ambiguity
with other providers:

```bash
llm -m "unsloth/Qwen3.8-27B-GGUF" "Write a haiku about GPUs"
```

The original Studio model ID is also registered as an alias, so this shorter
command selects the same model:

```bash
llm -m "unsloth/Qwen3.8-27B-GGUF" "Write a haiku about GPUs"
```

When Studio exposes one model, the short `unsloth` alias is also available:

```bash
llm -m unsloth "Explain QLoRA in one paragraph"
```

The plugin supports streaming, conversations, image attachments, tools, JSON
schemas, async models, and token usage through LLM's OpenAI-compatible model
implementation.

> [!NOTE]
> A model appearing in `llm models list` means Studio exposes it through
> `GET /v1/models`. It may not be loaded for inference yet. Start Studio with
> `unsloth run --model ...`, load it in the UI, or enable model auto-switch.

<!-- Separate adjacent GitHub callouts for Markdown linting. -->

> [!WARNING]
> Cloudflare quick tunnels do not preserve server-sent events. Use
> `llm --no-stream` when `UNSLOTH_STUDIO_URL` points to a quick tunnel.

## Development

Install dependencies and run the checks with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
