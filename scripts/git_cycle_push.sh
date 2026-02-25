#!/usr/bin/env bash
set -euo pipefail
msg="${1:-chore: update metrics}"
git add -A
git commit -m "$msg" || true
git push --force-with-lease origin main
