#!/usr/bin/env bash
set -euo pipefail

# Push ./output/* to a result branch on origin. Creates branch if it doesn't exist.
# Env overrides:
#   RESULT_BRANCH: branch name to push to (default: results-<YYYYmmdd>-<short_sha|manual>)
#   GIT_USER_NAME, GIT_USER_EMAIL: git identity (defaults provided)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "output" ]; then
  echo "No output directory found. Nothing to push." >&2
  exit 0
fi

SHORT_SHA=${CI_COMMIT_SHORT_SHA:-${GITHUB_SHA:-${CI_COMMIT_SHA:-""}}}
DATE_STR=$(date +%Y%m%d)
DEFAULT_BRANCH="results-${DATE_STR}-${SHORT_SHA:-manual}"
RESULT_BRANCH=${RESULT_BRANCH:-$DEFAULT_BRANCH}

GIT_USER_NAME=${GIT_USER_NAME:-"tabformer-ci"}
GIT_USER_EMAIL=${GIT_USER_EMAIL:-"tabformer-ci@example.com"}

git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
git config --global --add safe.directory "$ROOT_DIR"

git fetch --all --prune

# If a push URL is provided (e.g., token-authenticated HTTPS), use it
if [ -n "${GIT_PUSH_URL:-}" ]; then
  echo "Using custom push URL for origin"
  git remote set-url origin "$GIT_PUSH_URL"
fi

if git ls-remote --exit-code --heads origin "$RESULT_BRANCH" >/dev/null 2>&1; then
  echo "Remote branch exists: origin/$RESULT_BRANCH"
  git checkout -B "$RESULT_BRANCH" "origin/$RESULT_BRANCH"
else
  echo "Creating new branch: $RESULT_BRANCH"
  git checkout -B "$RESULT_BRANCH"
fi

# Stage outputs
git add -A output/

# Commit if there are changes
if git diff --cached --quiet; then
  echo "No changes to commit in ./output. Skipping push."
  exit 0
fi

COMMIT_MSG="ci: add training outputs on ${DATE_STR} (${SHORT_SHA:-local})"
git commit -m "$COMMIT_MSG"
git push -u origin "$RESULT_BRANCH"
echo "Pushed results to origin/$RESULT_BRANCH"
