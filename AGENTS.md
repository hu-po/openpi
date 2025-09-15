# Repository Guidelines

## Project Structure & Modules
- `src/openpi/`: Core library code (models, training, transforms). Tests live alongside modules as `*_test.py`.
- `packages/openpi-client/`: Client SDK used by examples and runtime.
- `scripts/`: Entry points for training, serving, utilities (e.g., `train.py`, `serve_policy.py`).
- `examples/`: Runnable demos and datasets (e.g., `examples/droid/`, `examples/aloha_*`, `examples/simple_client/`).
- `docs/`: How‑tos and deployment notes; see `docs/docker.md`, `docs/remote_inference.md`.
- `third_party/`: External code and assets.

## Build, Test, and Dev Commands
- Install toolchain: `uv python install` (ensures Python), then `uv sync --all-extras --dev` (installs deps + dev group).
- Run tests: `uv run pytest --strict-markers -m "not manual"` (skips long/manual tests by default).
- Lint/format: `uv run ruff check . --fix` and `uv run ruff format .`.
- Pre-commit: `pre-commit install` then `pre-commit run -a` (uses uv-managed env).
- Example run: `uv run python scripts/serve_policy.py --help` or `uv run python examples/simple_client/main.py`.

## Coding Style & Naming
- Python 3.11+, 4‑space indentation, type hints encouraged (`src/openpi/py.typed`).
- Line length 120; enforced by Ruff. Prefer PEP8: `snake_case` functions/modules, `PascalCase` classes, `UPPER_CASE` constants.
- Imports sorted via Ruff isort settings (single‑line third‑party imports).

## Testing Guidelines
- Framework: Pytest. Test paths: `src`, `scripts`, `packages`.
- Naming: `*_test.py` files and `test_*` functions; mark long/integration as `@pytest.mark.manual`.
- Quick run: `uv run pytest -q -m "not manual"`. Add focused tests near the code they validate.

## Commit & Pull Requests
- Commits: Use clear, imperative titles (e.g., "Add SIGLIP loader error checks"). Group related changes; keep diffs small.
- PRs: Describe motivation, approach, and testing. Link issues/discussions. Include commands/logs, screenshots when UI‑like, and note any docs updates.
- Quality gate: CI runs `uv sync --all-extras --dev`, Ruff, and Pytest. Ensure all pass locally before opening a PR.

## Security & Configuration Tips
- GPU‑related and media deps may be required (FFmpeg, CUDA). See Dockerfiles in `scripts/docker/` and example `compose.yml` files for reproducible envs.
- Avoid committing large artifacts; use links or datasets referenced in `examples/` and `docs/`.
