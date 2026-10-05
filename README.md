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

The plugin discovers chat models from `GET /v1/models` when LLM starts. It also
registers the Decision API as `unsloth:decision` because Decision API models are
served separately from the OpenAI-compatible model list. List all models:

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

## Decision models

Unsloth now supports decision models. See their documentation [here](https://unsloth.ai/docs/models/decision-laya). You might need to check if your version of Unsloth already supports decision models. If you don't see this option, upgrade to the latest version of Unsloth.

In Unsloth Studio, open **Settings > API > Decision API** and turn on **Serve
requests**. The plugin always lists `unsloth:decision`, even when the Decision
API is disabled, because Studio does not provide a discovery endpoint for
`/v1/systemone`.

Pass a complete Decision API request as JSON through standard input:

```powershell
@'
{
  "model": "laya",
  "state": "I was charged twice. Please refund the duplicate today.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, errors",
        "other": "everything else"
      }
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the customer ask for a refund?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "today"]
    }
  }
}
'@ | llm -m unsloth-decision
```

For a text state, provide the questions as a model option:

```powershell
$questions = '{"refund":{"type":"noul","instructions":"Is a refund requested?"}}'
llm -m unsloth-decision `
  -o questions $questions `
  "I was charged twice. Please refund the duplicate."
```

The default Decision API model is `laya`. Select another model configured in
Studio with `-o decision_model laya-multilingual`. Supported values include
`default`, `jev-latest`, `laya-multilingual`, `laya-english`, and
`laya-typed-decisions`.

The Python API accepts the questions as a dictionary:

```python
import llm

model = llm.get_model("unsloth-decision")
response = model.prompt(
    "I was charged twice. Please refund the duplicate.",
    questions={
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
        }
    },
)
print(response.text())
```

Decision API responses are emitted as JSON. Their `input_tokens` and
`output_tokens` values are also available through `response.usage()`.

> [!NOTE]
> A chat model appearing in `llm models list` means Studio exposes it through
> `GET /v1/models`. It may not be loaded for inference yet. The
> `unsloth:decision` entry is registered independently and requires **Serve
> requests** under **Settings > API > Decision API**.

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
