#!/usr/bin/env python3
"""Sanity checks for OmniSTEval instances.resegmented.jsonl emissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml


def _load_instances(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_offsets_ms(path: Path) -> list[float]:
    with path.open(encoding="utf-8") as f:
        segs = yaml.safe_load(f)
    if not isinstance(segs, list):
        raise ValueError(f"Expected YAML list in {path}")
    return [float(seg["offset"]) * 1000.0 for seg in segs]


def _iter_docs(rows: list[dict]) -> Iterable[tuple[int, list[dict]]]:
    docs: dict[int, list[dict]] = {}
    for row in rows:
        docs.setdefault(int(row.get("docid", 0)), []).append(row)
    for docid in sorted(docs):
        yield docid, sorted(docs[docid], key=lambda x: int(x.get("segid", x["index"])))


def analyze(rows: list[dict], offsets_ms: list[float], ts_key: str) -> dict:
    total_tokens = 0
    negative_relative = 0
    within_segment_nonmonotonic = 0
    cross_segment_backwards_absolute = 0
    docs_checked = 0

    for _, doc_rows in _iter_docs(rows):
        docs_checked += 1
        prev_abs_last = None
        for row in doc_rows:
            idx = int(row["index"])
            rel = row.get(ts_key, []) or []
            if not rel:
                continue

            total_tokens += len(rel)
            negative_relative += sum(1 for x in rel if x < 0.0)
            within_segment_nonmonotonic += sum(
                1 for i in range(1, len(rel)) if rel[i] < rel[i - 1]
            )

            abs_vals = [offsets_ms[idx] + float(x) for x in rel]
            if prev_abs_last is not None and abs_vals[0] < prev_abs_last:
                cross_segment_backwards_absolute += 1
            prev_abs_last = abs_vals[-1]

    return {
        "docs_checked": docs_checked,
        "total_tokens": total_tokens,
        "negative_relative": negative_relative,
        "within_segment_nonmonotonic": within_segment_nonmonotonic,
        "cross_segment_backwards_absolute": cross_segment_backwards_absolute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True, help="instances.resegmented.jsonl")
    parser.add_argument("--segmentation", type=Path, required=True, help="audio_definition.yaml")
    args = parser.parse_args()

    rows = _load_instances(args.instances)
    offsets_ms = _load_offsets_ms(args.segmentation)

    if not rows:
        raise SystemExit("No rows found in instances file.")
    if max(int(r["index"]) for r in rows) >= len(offsets_ms):
        raise SystemExit("Segmentation does not cover all instance indices.")

    print(f"Checked {len(rows)} segments from {args.instances}")
    print(f"Segmentation entries: {len(offsets_ms)} from {args.segmentation}")
    for ts_key in ("emission_cu", "emission_ca"):
        stats = analyze(rows, offsets_ms, ts_key=ts_key)
        print("")
        print(f"[{ts_key}]")
        print(f"  docs_checked: {stats['docs_checked']}")
        print(f"  total_tokens: {stats['total_tokens']}")
        print(f"  negative_relative: {stats['negative_relative']}")
        print(f"  within_segment_nonmonotonic: {stats['within_segment_nonmonotonic']}")
        print(
            "  cross_segment_backwards_absolute: "
            f"{stats['cross_segment_backwards_absolute']}"
        )


if __name__ == "__main__":
    main()
