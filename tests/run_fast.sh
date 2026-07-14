#!/usr/bin/env bash
# Fast subset: Tier 0 + Tier 1 smoke. Designed to run in under ~2 minutes on
# a single H100. Intended for per-commit CI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

pytest tests/unit tests/e2e -m "tier0 or smoke" -v "$@"
