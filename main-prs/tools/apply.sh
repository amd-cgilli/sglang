#!/usr/bin/env bash
# Rebuild qwen38-submission-v2 from upstream main: every main-prs/NN-*/sglang.patch in order, then the main-prs/ folder.
# Optionally rebuild the AITER tree: the commit SGLang's ROCm Dockerfile pins + every main-prs/NN-*/aiter.patch.
#
# usage: apply.sh [--sync] [--base <rev>] [--aiter <aiter-dir>] [--dry-run]
#   --sync      fetch upstream main first and rebuild on it (default base: the branch's current upstream base)
#   --base      rebuild on this commit instead
#   --aiter     also reset <aiter-dir> to the pinned AITER commit (detached) and apply the AITER patches
#   --dry-run   only check that every patch applies (or is already upstream)
# A patch that is already in the base (reverse-applies cleanly) is skipped and reported as merged.
# The first patch that does not apply stops the rebuild and is named; the branch is left untouched.
set -euo pipefail
PRS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(git -C "$PRS" rev-parse --show-toplevel)
UPSTREAM=${UPSTREAM_URL:-https://github.com/sgl-project/sglang.git}
AITER_UPSTREAM=${AITER_UPSTREAM_URL:-https://github.com/ROCm/aiter.git}
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
sync=0 base='' aiter='' dry=0
while (($#)); do
  case $1 in
    --sync) sync=1; shift ;;
    --base) base=$2; shift 2 ;;
    --aiter) aiter=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -z $(git -C "$REPO" status --porcelain) ]] || { echo "worktree not clean; commit or stash first" >&2; exit 1; }

if ((sync)); then
  git -C "$REPO" fetch -q "$UPSTREAM" main
  base=$(git -C "$REPO" rev-parse FETCH_HEAD)
elif [[ -z $base ]]; then
  base=$(cat "$PRS/BASE" 2>/dev/null || true)
  [[ -n $base ]] || { echo "no base: pass --base or --sync (main-prs/BASE is missing)" >&2; exit 1; }
fi
echo "base: $(git -C "$REPO" log -1 --format='%h %cd %s' --date=short "$base")"

tmp=$(mktemp -d); trap 'git -C "$REPO" worktree remove --force "$tmp/wt" 2>/dev/null; rm -rf "$tmp"' EXIT
cp -a "$PRS" "$tmp/main-prs"
git -C "$REPO" worktree add -q --detach "$tmp/wt" "$base"
merged=()
for p in "$tmp"/main-prs/[0-9]*/sglang.patch; do
  [[ -f $p ]] || continue
  name=$(basename "$(dirname "$p")")
  if git -C "$tmp/wt" apply --check -R "$p" 2>/dev/null; then
    echo "  $name: already in base (merged upstream) - skipped"; merged+=("$name"); continue
  fi
  if ! git -C "$tmp/wt" am -q -3 "$p"; then
    git -C "$tmp/wt" am --abort || true
    echo "STOP: $name does not apply on $base; rebase it (fix in a worktree on this base, then export.sh)" >&2; exit 1
  fi
  echo "  $name: applied"
done
rm -rf "$tmp/wt/main-prs"; cp -a "$tmp/main-prs" "$tmp/wt/main-prs"
git -C "$tmp/wt" rev-parse "$base" > "$tmp/wt/main-prs/BASE"
git -C "$tmp/wt" add main-prs && git -C "$tmp/wt" commit -qm "main-prs: tracking (patches, metadata, tools)"
((${#merged[@]})) && echo "merged upstream (consider marking status: merged): ${merged[*]}"

if ((dry)); then
  echo "dry run: all patches apply; branch $BRANCH unchanged"
else
  new=$(git -C "$tmp/wt" rev-parse HEAD)
  git -C "$REPO" reset -q --hard "$new"
  echo "$BRANCH rebuilt at $(git -C "$REPO" log -1 --format=%h) on $(git -C "$REPO" rev-parse --short "$base")"
fi

if [[ -n $aiter ]]; then
  pin=$(git -C "$REPO" show "$base:docker/rocm.Dockerfile" | grep -oE 'AITER_COMMIT_DEFAULT="[0-9a-f]+"' | grep -oE '[0-9a-f]{40}' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
  echo "aiter pin (most common AITER_COMMIT_DEFAULT): $pin"
  git -C "$aiter" cat-file -e "$pin^{commit}" 2>/dev/null || git -C "$aiter" fetch -q "$AITER_UPSTREAM" "$pin"
  ((dry)) && { echo "dry run: aiter not touched"; exit 0; }
  [[ -z $(git -C "$aiter" status --porcelain --untracked-files=no) ]] || { echo "aiter tree not clean" >&2; exit 1; }
  git -C "$aiter" checkout -q --detach "$pin"
  for p in "$PRS"/[0-9]*/aiter.patch; do
    [[ -f $p ]] || continue
    name=$(basename "$(dirname "$p")")
    if git -C "$aiter" apply --check -R "$p" 2>/dev/null; then echo "  $name (aiter): already in pin - skipped"; continue; fi
    git -C "$aiter" am -q -3 "$p" || { git -C "$aiter" am --abort || true; echo "STOP: $name (aiter) does not apply on $pin" >&2; exit 1; }
    echo "  $name (aiter): applied"
  done
  git -C "$aiter" submodule update --init 3rdparty/composable_kernel >/dev/null
  echo "aiter at $(git -C "$aiter" log -1 --format=%h) (pin $pin + patches)"
fi
