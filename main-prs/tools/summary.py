#!/usr/bin/env python3
"""Regenerate the summary table in main-prs/README.md from every main-prs/NN-*/meta.yaml."""

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- summary:begin -->", "<!-- summary:end -->"


def _stats(meta: dict) -> str:
    parts = []
    for repo in ("sglang", "aiter"):
        s = (meta.get("changes") or {}).get(repo)
        if s:
            parts.append(f"{repo}: {s['files']} files, +{s['insertions']} / -{s['deletions']}")
    return "<br>".join(parts) or "-"


def _impact(meta: dict) -> str:
    imp = meta.get("impact") or {}
    if not imp:
        return "not measured"
    text = imp.get("summary", "-")
    if imp.get("measured"):
        text += f" ({imp['measured']})"
    return text


def main() -> int:
    rows = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        meta = yaml.safe_load((d / "meta.yaml").read_text())
        base = meta.get("base") or {}
        rows.append(
            f"| [{d.name}]({d.name}/) | {meta.get('title', '')} | {meta.get('status', 'local')} "
            f"| {_stats(meta)} | {_impact(meta)} "
            f"| {base.get('sglang', '-')} / {base.get('aiter', '-')} |"
        )
    table = "\n".join(
        [
            "| PR | What | Status | Changes | Measured impact | Base (sglang main / aiter) |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )
    readme = ROOT / "README.md"
    text = readme.read_text()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{BEGIN}\n{table}\n{END}{tail}")
    print(f"{len(rows)} PRs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
