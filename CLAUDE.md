# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Setup and commands

This is a `uv` + `mise` managed Python project (Python >=3.13). All tasks run through `mise` (see `mise.toml`), which wraps `uv`:

```bash
mise run dev              # install prod + dev deps (runs `clean` first)
mise run install          # install prod deps only (uv sync --no-dev)
mise run run              # run the Streamlit app (src/markdowns_organizer/app.py)
mise run lint              # ruff check on src/ and tests/
mise run format            # ruff check --fix, pyupgrade --py313-plus, ruff format
mise run test              # uv run pytest tests/ -q  (note: task appends `|| true`, so it never fails the shell)
mise run update            # uv lock --upgrade
mise run clean             # remove .venv and cached files
```

To run a single test or invoke pytest/ruff directly, use `uv run` (the project venv is managed by uv):

```bash
uv run pytest tests/path_to_test.py::test_name
uv run ruff check src/markdowns_organizer/app.py
```

Graphify maintenance tasks (`graphify-install`, `graphify-uninstall`, `graphify-refresh`, `graphify-viz`) are also defined in `mise.toml`.

## Architecture

The installable package is `src/markdowns_organizer/`, a single-file Streamlit app (`app.py`) named "Markdown Organizer". It exposes one helper, `save_markdown(data, folder)`, which writes text to `<folder>/data.md` (creating the folder if needed), wired to a minimal Streamlit UI (text area + folder name input + save button). `tests/` currently has no test modules beyond `__init__.py`.

`pyproject.toml` configures Ruff with `select = ["ALL"]` (all lint rules enabled) and only a few rules ignored (`E501`, `CPY001`, `D100`, `D104`) — new code should expect strict linting by default rather than relying on suppressions.

The rest of the repository is non-code content, not part of the Python package:
- `coursera/machine_learning/` — Coursera ML course notebooks and exercises (`.ipynb`, data files, plots), organized by course/week (`c1w1`, `c2w3`, etc.), each with `practice`/`exam` subfolders.
- `research_topics/` — markdown research notes on topics like agents, LLM hosting, and multi-platform models, often with parallel writeups from different AI tools (e.g. `claud.md`, `gemini.md`, `grock.md` for the same prompt).
