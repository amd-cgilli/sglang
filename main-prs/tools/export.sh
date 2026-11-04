#!/usr/bin/env bash
# Export one PR's patch(es) and change stats into main-prs/<NN-name>/ from commits.
# usage: export.sh <NN-name> [--sglang <rev-range>] [--aiter <aiter-dir> <rev-range>]
#   e.g. export.sh 01-qsa-decode-indexer --sglang abc123~1..abc123
#        export.sh 03-gdn-packed --sglang a1~1..a1 --aiter /sgl-workspace/aiter-v2 b2~1..b2
# Writes sglang.patch / aiter.patch (git format-patch mbox) and updates meta.yaml:
# changes.<repo> (files / insertions / deletions) and base.<repo> (the upstream commit the patch was made on).
set -euo pipefail
PRS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=$1; shift
DIR=$PRS/$NAME
mkdir -p "$DIR"
[[ -f $DIR/meta.yaml ]] || printf 'title: %s\nstatus: local\ndescription: |\n  TODO\nimpact: {}\n' "$NAME" > "$DIR/meta.yaml"

export_repo() {  # <repo label> <git dir> <rev-range>
  local label=$1 gdir=$2 range=$3
  git -C "$gdir" format-patch --stdout "$range" > "$DIR/$label.patch"
  read -r files ins del < <(git -C "$gdir" diff --shortstat "${range%%..*}" "${range##*..}" \
    | awk '{f=$1; i=0; d=0; for (k=1; k<=NF; k++) { if ($(k)~/insertion/) i=$(k-1); if ($(k)~/deletion/) d=$(k-1) } print f, i, d}')
  # base = the upstream commit under the stack (main-prs/BASE; for AITER the pin SGLang's Dockerfile names at BASE),
  # not the previous PR's commit. Falls back to the range start when that commit is not one of its ancestors.
  local start=${range%%..*} up='' sgl_base base
  sgl_base=$(cat "$PRS/BASE" 2>/dev/null || true)
  if [[ $label == sglang ]]; then up=$sgl_base
  elif [[ -n $sgl_base ]]; then
    up=$(git -C "$PRS" show "$sgl_base:docker/rocm.Dockerfile" 2>/dev/null | grep -oE 'AITER_COMMIT_DEFAULT="[0-9a-f]+"' \
      | grep -oE '[0-9a-f]{40}' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}' || true)
  fi
  if [[ -n $up ]] && git -C "$gdir" merge-base --is-ancestor "$up" "$start" 2>/dev/null; then start=$up; fi
  base=$(git -C "$gdir" rev-parse --short=10 "$start")
  python3 - "$DIR/meta.yaml" "$label" "$files" "$ins" "$del" "$base" <<'PY'
import sys, yaml
path, label, files, ins, dele, base = sys.argv[1:]
meta = yaml.safe_load(open(path)) or {}
meta.setdefault("changes", {})[label] = {"files": int(files), "insertions": int(ins), "deletions": int(dele)}
meta.setdefault("base", {})[label] = base
yaml.safe_dump(meta, open(path, "w"), sort_keys=False, width=110)
PY
  echo "$NAME/$label.patch: $files files, +$ins / -$del (base $base)"
}

while (($#)); do
  case $1 in
    --sglang) export_repo sglang "$(git -C "$PRS" rev-parse --show-toplevel)" "$2"; shift 2 ;;
    --aiter) export_repo aiter "$2" "$3"; shift 3 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
python3 "$PRS/tools/summary.py"
