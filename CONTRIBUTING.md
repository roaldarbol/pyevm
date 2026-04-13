# Contributing to pyevm

Thank you for your interest in contributing! Here is everything you need to get started.

## Development setup

pyevm uses [pixi](https://pixi.sh) to manage the development environment. It handles
Python, PyTorch, and (on Windows/Linux) CUDA-accelerated torchcodec in one step.

```bash
# Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | sh

# Clone and set up
git clone https://github.com/roaldarbol/pyevm.git
cd pyevm
pixi install

# Install git hooks (ruff format/lint on commit, pytest on push)
pixi run lefthook install
```

## Running tests

```bash
pixi run test
```

Coverage report is printed to the terminal; an XML report is written to `coverage.xml`
for CI upload.

## Linting and formatting

```bash
pixi run lint      # ruff check
pixi run format    # ruff format
```

The pre-commit hook runs both automatically on `git commit`.

## Security audit

```bash
pixi run audit     # pip-audit
```

## Building docs locally

```bash
pixi run docs-serve   # live-reload preview at http://localhost:8000
pixi run docs         # one-shot build into site/
```

## Submitting changes

1. Open an issue first for non-trivial changes so we can discuss the approach.
2. Fork the repository and create a feature branch.
3. Make your changes and add/update tests.
4. Ensure `pixi run test` and `pixi run lint` both pass.
5. Open a pull request against `main`.

## Releasing (maintainers only)

Releases are triggered by pushing a version tag. hatch-vcs picks up the tag
automatically — no manual version bumping required.

```bash
git tag v1.2.3
git push origin v1.2.3
```

The `release` GitHub Actions workflow builds the wheel and sdist and publishes
to PyPI via OIDC Trusted Publisher (no API token stored).

## Code style

- Python 3.12+, formatted with **ruff** (line length 100).
- Type annotations on all public functions.
- Google-style docstrings (used by mkdocstrings for API docs).
