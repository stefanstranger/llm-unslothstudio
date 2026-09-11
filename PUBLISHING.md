---
title: Publishing llm-unslothstudio to PyPI
description: Step-by-step uv-first release guide for building, validating, and publishing the llm-unslothstudio package to PyPI.
author: stefanstranger
ms.date: 2026-09-11
ms.topic: how-to
keywords:
  - uv
  - pypi
  - python packaging
  - twine
  - release
estimated_reading_time: 8
---

## Overview

This guide shows how to publish this package with uv.

Project name on PyPI: llm-unslothstudio.

## Prerequisites

* A PyPI account
* uv installed and available in your terminal
* Access to create a PyPI API token
* Terminal opened at repository root

Check your current folder:

```powershell
Get-Location
```

Expected path:

```text
C:\github\llm-unslothstudio
```

## 1) Install and verify uv

If uv is not installed yet, use one of the install methods at:

<https://docs.astral.sh/uv/getting-started/installation/>

Verify:

```powershell
uv --version
```

## 2) Create and activate a virtual environment

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

Install packaging tools in the virtual environment:

```powershell
uv pip install --upgrade build twine
```

## 3) Clean old build artifacts

```powershell
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
if (Test-Path build) { Remove-Item build -Recurse -Force }
Get-ChildItem -Directory -Filter *.egg-info | Remove-Item -Recurse -Force
```

## 4) Build distributions

```powershell
uv run python -m build
```

Confirm dist files exist:

```powershell
Get-ChildItem dist
```

Expected files:

* One wheel file ending with .whl
* One source distribution ending with .tar.gz

## 5) Validate metadata and README

Run Twine checks:

```powershell
uv run python -m twine check .\dist\*
```

Expected result:

```text
PASSED
```

## 6) Create a PyPI token

Create a token at:

<https://pypi.org/manage/account/token/>

For first publication of a new project name, use an account-scoped token.
After first successful publish, create and use a project-scoped token.

## 7) Publish with uv

### Option A: Interactive token prompt

```powershell
uv publish
```

When prompted:

* Username: __token__
* Password: paste your PyPI token value

### Option B: Temporary environment variable

```powershell
$env:UV_PUBLISH_TOKEN="pypi-REPLACE_WITH_YOUR_TOKEN"
uv publish
Remove-Item Env:UV_PUBLISH_TOKEN
```

## Automated releases with GitHub Actions

The [publish workflow](.github/workflows/publish.yml) runs when a GitHub Release
is published and can also be started manually. It builds and checks the
distributions in a separate job, then publishes them through PyPI Trusted
Publishing without a stored API token.

Before using the workflow:

1. Create a `pypi` GitHub Environment in the repository settings, optionally
   with required reviewers.
2. Register this repository and `.github/workflows/publish.yml` as a trusted
   publisher at <https://pypi.org/manage/account/publishing/>.

If Trusted Publishing is not configured yet, use the manual token process
above. Do not add a token to the workflow unless it is passed through a GitHub
Actions secret such as `PYPI_API_TOKEN`.

## 8) Verify release

Open the project page:

<https://pypi.org/project/llm-unslothstudio/>

Test install:

```powershell
uv pip install llm-unslothstudio
```

Or with pip:

```powershell
python -m pip install llm-unslothstudio
```

Verify that LLM can install and discover the plugin:

```powershell
llm install llm-unslothstudio
llm plugins
```

The `llm plugins` output should include `llm-unslothstudio`.

## 9) Future releases

Bump version in [pyproject.toml](pyproject.toml), then repeat:

* Clean artifacts
* Build
* Twine check
* Publish

Important PyPI rule: you cannot overwrite an already uploaded version. Every
release must use a new version.

## License

The package declares `license = "Apache-2.0"` in
[pyproject.toml](pyproject.toml), but the repository previously had no license
file. A standard Apache-2.0 [LICENSE](LICENSE) is now included at the repository
root so that the declared license text ships with the package. Keep this file in
future distributions; declaring a license without its text is inconsistent.

## Troubleshooting

### twine check exits with code 1

Common causes:

* dist folder is missing because build did not run
* dist folder exists but is empty
* PowerShell wildcard matched no files

Quick recovery:

```powershell
uv run python -m build
Get-ChildItem dist
uv run python -m twine check .\dist\*
```

### publish fails with authentication error

* Confirm token starts with pypi-
* Confirm token was copied fully
* Confirm token scope includes this project
* Try interactive mode with `uv publish` and paste the token directly
