#!/usr/bin/env bash
# Uploads this project to a Hugging Face Space (Docker SDK).
#
# Only uploads files git would track (tracked or untracked but not .gitignore'd),
# so .venv, caches and logs never get sent. The hf CLI handles binary files (the
# .pkl models, the plot) on its own, so git-lfs isn't needed.
#
# One-time setup:  .venv/bin/pip install -r requirements-dev.txt
#                  .venv/bin/hf auth login
# Usage:           ./deploy_hf.sh <hf-username>/<space-name>
set -euo pipefail
cd "$(dirname "$0")"

SPACE="${1:?usage: ./deploy_hf.sh <hf-username>/<space-name>}"
HF="${HF:-.venv/bin/hf}"

"$HF" repos create "$SPACE" --repo-type space --space-sdk docker --exist-ok

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' f; do
    [ -e "$f" ] && cp --parents -- "$f" "$STAGE"
done

# --delete removes files from the Space that no longer exist locally.
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --delete "*" --commit-message "Deploy from local"
echo ""
echo "Deployed. The Space builds in a minute or two: https://huggingface.co/spaces/$SPACE"
