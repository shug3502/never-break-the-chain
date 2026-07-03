# celltrack task runner — run `just` to list recipes.
# Recipes use the project's .venv (see CLAUDE.md / AGENTS.md conventions).

# Python interpreter/tools from the local virtual environment.
python := ".venv/bin/python"
ruff := ".venv/bin/ruff"
pytest := python + " -m pytest"

# Show available recipes (default).
default:
    @just --list

# Install the package with dev dependencies into .venv.
install:
    {{python}} -m pip install -e ".[dev]"

# Run the test suite.
test *args:
    {{pytest}} -q {{args}}

# Lint with ruff.
lint:
    {{ruff}} check src tests

# Auto-fix lint issues where possible.
lint-fix:
    {{ruff}} check --fix src tests

# Check formatting without modifying files.
format-check:
    {{ruff}} format --check src tests

# Format code in place.
format:
    {{ruff}} format src tests

# Lint + format-check + tests: the full pre-commit gate.
check: lint format-check test
