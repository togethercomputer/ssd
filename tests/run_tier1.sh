#!/usr/bin/env bash
# Full Tier 1 suite: all single-GPU E2E tests. Takes ~8-10 minutes on H100.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

pytest tests/unit tests/e2e -m "tier0 or tier1" -v "$@"
