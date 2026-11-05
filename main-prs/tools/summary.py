#!/usr/bin/env python3
"""Regenerate the summary table in main-prs/README.md from every main-prs/NN-*/meta.yaml.

--refresh: read the state (draft / open / merged / closed) of our PRs (`prs:`) and of related upstream PRs (`related:`)
from the GitHub API into meta.yaml first.
"""

import json
import os
import pathlib
import subprocess
import sys
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- summary:begin -->", "<!-- summary:end -->"
RANK_BEGIN, RANK_END = "<!-- ranking:begin -->", "<!-- ranking:end -->"
BADGE = {"draft": "lightgrey", "open": "yellow", "merged": "brightgreen", "closed": "red"}


def _pr_state(url: str) -> str:
    tok = os.environ.get("GITHUB_TOKEN") or subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", capture_output=True, text=True
    ).stdout.partition("password=")[2].split("\n")[0]
    owner, repo, _, num = url.split("github.com/")[1].split("/")[:4]
    req = urllib.request.Request(f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.load(r)
    return "merged" if j.get("merged_at") else "closed" if j["state"] == "closed" else "draft" if j.get("draft") else "open"


TAIL = ("ranking", "pr_state", "related")  # kept at the end of meta.yaml (ranking: hand-set; the rest generated)


def write_tail(f: pathlib.Path, meta: dict) -> None:
    """Rewrite only the generated keys at the end of meta.yaml; the hand-written part stays byte-identical."""
    head = []
    for line in f.read_text().splitlines(keepends=True):
        if line.split(":")[0] in TAIL:
            break
        head.append(line)
    tail = {k: meta[k] for k in TAIL if meta.get(k)}
    f.write_text("".join(head) + (yaml.safe_dump(tail, sort_keys=False, width=110) if tail else ""))


def link(url: str, state: str) -> str:
    """[repo#N](url) + colored state badge."""
    owner_repo, _, num = url.split("github.com/")[1].rsplit("/", 2)
    return f"[{owner_repo.split('/')[1]}#{num}]({url}) ![{state}](https://img.shields.io/badge/-{state}-{BADGE.get(state, 'lightgrey')})"


def pr_cell(meta: dict) -> str:
    """Our PR(s), or 'no PR'."""
    return "<br>".join(link(u, (meta.get("pr_state") or {}).get(r, "open")) for r, u in (meta.get("prs") or {}).items()) or "no PR"


def related_cell(meta: dict) -> str:
    """Open upstream PRs doing the same change, an alternative implementation, or a prerequisite."""
    return "<br>".join(f"{x['relation']}: {link(x['url'], x['state'])}" for x in meta.get("related") or []) or "none found"


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


def _lines(meta: dict) -> int:
    return sum(c["insertions"] + c["deletions"] for c in (meta.get("changes") or {}).values())


def ranking_table(metas: dict) -> str:
    """Changes by agentic C24 P90 contribution; a change's prerequisites (ranking.requires) are folded into its row."""
    rows = sorted(((n, m) for n, m in metas.items() if "p90_c24" in (m.get("ranking") or {})),
                  key=lambda nm: -nm[1]["ranking"]["p90_c24"])
    out = ["| Rank | Change | P90 (tok/s/user) | tput (tok/s/GPU) | Within noise | Lines (+/-) | GitHub PR | Same change upstream |",
           "|---|---|---|---|---|---|---|---|"]
    for i, (n, m) in enumerate(rows, 1):
        r = m["ranking"]
        parts = [n] + [k for k in metas if k[:2] in r.get("requires", [])]
        same = "<br>".join(dict.fromkeys(link(x["url"], x["state"]) for p in parts for x in metas[p].get("related") or []
                                         if x["relation"] in ("same change", "prerequisite"))) or "none"
        ours = "<br>".join(c for c in dict.fromkeys(pr_cell(metas[p]) for p in parts) if c != "no PR") or "no PR"
        name = f"[{n}]({n}/)" + "".join(f" + [{p[:2]}]({p}/)" for p in parts[1:])
        note = f"<br>{r['note']}" if r.get("note") else ""
        out.append(f"| {i} | {name} | {r['p90_c24']:+.1f}{note} | {r['tput_c24_k']:+.1f}k | {'yes' if r['noise'] else '**no**'} "
                   f"| {sum(_lines(metas[p]) for p in parts)} | {ours} | {same} |")
    return "\n".join(out)


def _splice(text: str, begin: str, end: str, body: str) -> str:
    head, rest = text.split(begin, 1)
    return f"{head}{begin}\n{body}\n{end}{rest.split(end, 1)[1]}"


def main() -> int:
    rows, metas = [], {}
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        meta = yaml.safe_load((d / "meta.yaml").read_text())
        if "--refresh" in sys.argv and (meta.get("prs") or meta.get("related")):
            if meta.get("prs"):
                meta["pr_state"] = {repo: _pr_state(url) for repo, url in meta["prs"].items()}
            for x in meta.get("related") or []:
                x["state"] = _pr_state(x["url"])
            write_tail(d / "meta.yaml", meta)
        metas[d.name] = meta
        base = meta.get("base") or {}
        rows.append(
            f"| [{d.name}]({d.name}/) | {meta.get('title', '')} | {pr_cell(meta)} | {related_cell(meta)} "
            f"| {_stats(meta)} | {_impact(meta)} "
            f"| {base.get('sglang', '-')} / {base.get('aiter', '-')} |"
        )
    table = "\n".join(
        [
            "| PR | What | GitHub PR | Related upstream (open) | Changes | Measured impact | Base (sglang main / aiter) |",
            "|---|---|---|---|---|---|---|",
            *rows,
        ]
    )
    readme = ROOT / "README.md"
    text = _splice(readme.read_text(), BEGIN, END, table)
    if RANK_BEGIN in text:
        text = _splice(text, RANK_BEGIN, RANK_END, ranking_table(metas))
    readme.write_text(text)
    print(f"{len(rows)} PRs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
