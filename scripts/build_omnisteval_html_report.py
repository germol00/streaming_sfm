#!/usr/bin/env python3
"""Build a browsable HTML report from OmniSTEval instances.resegmented.jsonl."""

from __future__ import annotations

import argparse
import html
import json
from difflib import SequenceMatcher
from pathlib import Path


def _word_diff_html(reference: str, prediction: str) -> str:
    ref_tokens = reference.split()
    hyp_tokens = prediction.split()
    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens)
    ref_parts: list[str] = []
    hyp_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = " ".join(html.escape(w) for w in ref_tokens[i1:i2])
        hyp_chunk = " ".join(html.escape(w) for w in hyp_tokens[j1:j2])
        if tag == "equal":
            ref_parts.append(ref_chunk)
            hyp_parts.append(hyp_chunk)
        elif tag == "delete":
            ref_parts.append(f'<span class="del">{ref_chunk}</span>')
        elif tag == "insert":
            hyp_parts.append(f'<span class="ins">{hyp_chunk}</span>')
        elif tag == "replace":
            ref_parts.append(f'<span class="sub">{ref_chunk}</span>')
            hyp_parts.append(f'<span class="sub">{hyp_chunk}</span>')

    ref_line = " ".join(p for p in ref_parts if p)
    hyp_line = " ".join(p for p in hyp_parts if p)
    return (
        f'<div class="line"><span class="label">REF</span> {ref_line or "<em>(empty)</em>"}</div>'
        f'<div class="line"><span class="label">HYP</span> {hyp_line or "<em>(empty)</em>"}</div>'
    )


def _match_ratio(reference: str, prediction: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = prediction.split()
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return SequenceMatcher(None, ref_tokens, hyp_tokens).ratio()


def build_report(instances_path: Path, out_path: Path, title: str, max_instances: int) -> None:
    rows: list[dict] = []
    with instances_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    for row in rows:
        ref = (row.get("reference") or "").strip()
        hyp = (row.get("prediction") or "").strip()
        row["_ratio"] = _match_ratio(ref, hyp)
        row["_empty"] = len(hyp) == 0

    rows.sort(key=lambda r: (not r["_empty"], r["_ratio"]))
    if max_instances > 0:
        rows = rows[:max_instances]

    empty_count = sum(1 for r in rows if r["_empty"])
    body_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;}",
        "h1{margin-bottom:0.2rem;} .meta{color:#555;margin-bottom:1.5rem;}",
        ".seg{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0;}",
        ".seg h3{margin:0 0 0.6rem;font-size:1rem;}",
        ".line{margin:0.35rem 0;line-height:1.5;}",
        ".label{display:inline-block;width:2.5rem;font-weight:600;color:#444;}",
        ".del{background:#ffe0e0;text-decoration:line-through;}",
        ".ins{background:#e0ffe8;}",
        ".sub{background:#fff3cd;}",
        ".badge{display:inline-block;font-size:0.8rem;padding:0.1rem 0.45rem;border-radius:4px;margin-left:0.4rem;}",
        ".badge.warn{background:#fff3cd;}",
        ".badge.bad{background:#ffe0e0;}",
        ".filters{margin:1rem 0;}",
        "</style>",
        "<script>",
        "function filterSegs(mode){",
        "  document.querySelectorAll('.seg').forEach(el=>{",
        "    const empty=el.dataset.empty==='1';",
        "    const ratio=parseFloat(el.dataset.ratio||'1');",
        "    let show=true;",
        "    if(mode==='errors') show=ratio<1.0;",
        "    else if(mode==='empty') show=empty;",
        "    else if(mode==='worst') show=ratio<0.5;",
        "    el.style.display=show?'block':'none';",
        "  });",
        "}",
        "</script>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>Source: {html.escape(str(instances_path))} · "
        f"{len(rows)} segments shown · {empty_count} empty predictions</p>",
        '<div class="filters">'
        '<button onclick="filterSegs(\'all\')">All</button> '
        '<button onclick="filterSegs(\'errors\')">With errors</button> '
        '<button onclick="filterSegs(\'worst\')">Worst (match &lt; 50%)</button> '
        '<button onclick="filterSegs(\'empty\')">Empty hypotheses</button>'
        "</div>",
    ]

    for row in rows:
        idx = row.get("index", "?")
        docid = row.get("docid", "")
        segid = row.get("segid", "")
        ref = (row.get("reference") or "").strip()
        hyp = (row.get("prediction") or "").strip()
        ratio = row["_ratio"]
        badge = ""
        if row["_empty"]:
            badge = '<span class="badge bad">empty</span>'
        elif ratio < 0.5:
            badge = f'<span class="badge bad">{ratio:.0%} match</span>'
        elif ratio < 1.0:
            badge = f'<span class="badge warn">{ratio:.0%} match</span>'

        body_parts.append(
            f'<section class="seg" data-ratio="{ratio:.4f}" data-empty="{1 if row["_empty"] else 0}">'
            f"<h3>Segment {idx} (doc {docid}, seg {segid}){badge}</h3>"
            f"{_word_diff_html(ref, hyp)}"
            "</section>"
        )

    body_parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"Wrote HTML report: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="OmniSTEval phrase report")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Limit segments in the report (0 = all)",
    )
    args = parser.parse_args()
    build_report(args.instances, args.output, args.title, args.max_instances)


if __name__ == "__main__":
    main()
