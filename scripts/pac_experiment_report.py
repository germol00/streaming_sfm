#!/usr/bin/env python3
"""Aggregate PAC experiment outputs into paper-ready summaries and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


QUALITY_METRICS = ("COMET", "BLEU", "chrF")
LATENCY_METRICS = ("LongLAAL (CA)", "LongYAAL (CA)", "LongAL (CA)", "LongLAAL (CU)")
STABILITY_METRICS = ("normalized_erasure", "deletion_ratio", "revision_step_rate")
COMPUTE_METRICS = ("real_time_factor", "metrics_log_rtf", "mean_computation_time_s")


def repo_root_from_manifest(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Manifest must be a YAML mapping: {path}")
    return manifest


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def experiment_output_dir(
    repo_root: Path,
    manifest: dict[str, Any],
    experiment: dict[str, Any],
    eval_split: str = "eval",
) -> Path:
    output_dir = experiment.get("output_dir")
    if output_dir:
        base = resolve_repo_path(repo_root, output_dir)
    else:
        output_root = resolve_repo_path(repo_root, manifest.get("output_root", "output/pac_experiments"))
        base = output_root / experiment["id"]
    if eval_split not in {"eval", "test"}:
        base = base / eval_split
    return base


def resolve_eval_split(cli_value: str | None, manifest: dict[str, Any]) -> str:
    return cli_value or manifest.get("mcif_set") or manifest.get("acl_set") or manifest.get("set", "eval")


def resolve_acl_set(cli_value: str | None, manifest: dict[str, Any]) -> str:
    return resolve_eval_split(cli_value, manifest)


def read_scores(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def segment_latency_stdevs(instances_path: Path) -> dict[str, float]:
    """Per-direction segment-level stdev of LongYAAL/LAAL (CA) from OmniSTEval resegmentation."""
    if not instances_path.exists():
        return {}
    try:
        from omnisteval.data import Instance
        from omnisteval.scoring import LAALScorer, YAALScorer
    except ImportError:
        return {}

    rows = read_jsonl(instances_path)
    if not rows:
        return {}

    instances = [Instance.from_dict(row) for row in rows]
    scorers = (
        ("LongYAAL (CA) seg_stdev", YAALScorer(computation_aware=True, is_longform=True)),
        ("LongLAAL (CA) seg_stdev", LAALScorer(computation_aware=True)),
    )
    result: dict[str, float] = {}
    for metric_name, scorer in scorers:
        values: list[float] = []
        for ins in instances:
            delays = getattr(ins, "emission_ca", None)
            if not delays:
                continue
            score = scorer.compute(ins)
            if score is not None:
                values.append(score)
        if len(values) >= 2:
            result[metric_name] = statistics.stdev(values)
    return result


def metrics_log_diagnostics(metrics_log: Path) -> dict[str, float]:
    rows = [row for row in read_jsonl(metrics_log) if "total_audio_processed" in row]
    generated_counts = [len(row.get("generated_tokens", [])) for row in rows]
    deleted_counts = [len(row.get("deleted_tokens", [])) for row in rows]
    compute_times = [float(row.get("computation_time", 0.0) or 0.0) for row in rows]
    generated_total = float(sum(generated_counts))
    deleted_total = float(sum(deleted_counts))
    max_audio = max((float(row.get("total_audio_processed", 0.0) or 0.0) for row in rows), default=0.0)
    return {
        "chunks": float(len(rows)),
        "generated_tokens_total": generated_total,
        "deleted_tokens_total": deleted_total,
        "deletion_ratio": deleted_total / generated_total if generated_total else 0.0,
        "revision_step_rate": (
            sum(1 for count in deleted_counts if count) / len(rows) if rows else 0.0
        ),
        "emission_step_rate": (
            sum(1 for count in generated_counts if count) / len(rows) if rows else 0.0
        ),
        "mean_computation_time_s": statistics.fmean(compute_times) if compute_times else 0.0,
        "p50_computation_time_s": percentile(compute_times, 50),
        "p95_computation_time_s": percentile(compute_times, 95),
        "max_computation_time_s": max(compute_times, default=0.0),
        "total_computation_time_s": sum(compute_times),
        "metrics_log_rtf": sum(compute_times) / max_audio if max_audio else 0.0,
    }


def collect_results(
    repo_root: Path,
    manifest: dict[str, Any],
    selected_ids: set[str] | None,
    eval_split: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    summary_rows: list[dict[str, str]] = []
    diagnostics: dict[str, Any] = {}
    experiments = manifest.get("experiments", [])
    directions = manifest.get(
        "directions", ["en-de", "en-fr", "en-nl", "en-pt", "en-ru", "en-tr"]
    )

    for experiment in experiments:
        exp_id = experiment["id"]
        if selected_ids and exp_id not in selected_ids:
            continue
        out_dir = experiment_output_dir(repo_root, manifest, experiment, eval_split)
        label = experiment.get("label", exp_id)
        group = experiment.get("group", "")

        for row in read_scores(out_dir / "scores.tsv"):
            summary_rows.append(
                {
                    "experiment": exp_id,
                    "label": label,
                    "group": group,
                    "direction": row.get("direction", ""),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", ""),
                    "details": row.get("details", ""),
                    "output_dir": str(out_dir),
                }
            )

        diagnostics[exp_id] = {
            "label": label,
            "group": group,
            "output_dir": str(out_dir),
            "directions": {},
        }
        for direction in directions:
            diag = metrics_log_diagnostics(out_dir / f"metrics_{direction}.jsonl")
            seg_stdevs = segment_latency_stdevs(
                out_dir / f"omnisteval_{direction}" / "instances.resegmented.jsonl"
            )
            diagnostics[exp_id]["directions"][direction] = {**diag, **seg_stdevs}
            for metric, value in diag.items():
                summary_rows.append(
                    {
                        "experiment": exp_id,
                        "label": label,
                        "group": group,
                        "direction": direction,
                        "metric": metric,
                        "value": f"{value:.6f}",
                        "details": "metrics_log_diagnostics",
                        "output_dir": str(out_dir),
                    }
                )
            for metric, value in seg_stdevs.items():
                summary_rows.append(
                    {
                        "experiment": exp_id,
                        "label": label,
                        "group": group,
                        "direction": direction,
                        "metric": metric,
                        "value": f"{value:.6f}",
                        "details": "segment_latency_stdev",
                        "output_dir": str(out_dir),
                    }
                )

    return summary_rows, diagnostics


def write_summary_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "label", "group", "direction", "metric", "value", "details", "output_dir"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, ValueError):
            continue
        lookup[(row["experiment"], row["direction"], row["metric"])] = value
    return lookup


def mean_metric(lookup: dict[tuple[str, str, str], float], experiment: str, metric: str) -> float | None:
    values = [value for (exp, _direction, met), value in lookup.items() if exp == experiment and met == metric]
    if not values:
        return None
    return statistics.fmean(values)


def choose_metric(lookup: dict[tuple[str, str, str], float], experiment: str, candidates: tuple[str, ...]) -> tuple[str, float] | None:
    for metric in candidates:
        value = mean_metric(lookup, experiment, metric)
        if value is not None:
            return metric, value
    return None


def svg_scatter(path: Path, title: str, points: list[dict[str, Any]], x_label: str, y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 50, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [p["x"] for p in points if math.isfinite(p["x"])]
    ys = [p["y"] for p in points if math.isfinite(p["y"])]
    if not xs or not ys:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def sx(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - ((y - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="14">{y_label}</text>',
    ]
    for point in points:
        x, y = sx(point["x"]), sy(point["y"])
        label = point["label"]
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="black"/>')
        parts.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_bar(path: Path, title: str, bars: list[dict[str, Any]], y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 560
    left, right, top, bottom = 90, 30, 50, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max((bar["value"] for bar in bars), default=0.0)
    if max_value <= 0:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    slot = plot_w / max(1, len(bars))
    bar_w = slot * 0.65
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="14">{y_label}</text>',
    ]
    for idx, bar in enumerate(bars):
        h = (bar["value"] / max_value) * plot_h
        x = left + idx * slot + (slot - bar_w) / 2
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="black"/>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 5:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{bar["value"]:.3f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 18:.2f}" text-anchor="end" transform="rotate(-35 {x + bar_w / 2:.2f} {top + plot_h + 18:.2f})" font-family="sans-serif" font-size="11">{bar["label"]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lookup = metric_lookup(rows)
    experiments = [
        exp for exp in manifest.get("experiments", []) if exp.get("enabled", True)
    ]
    quality_points = []
    stability_points = []
    compute_bars = []
    for exp in experiments:
        exp_id = exp["id"]
        label = exp.get("label", exp_id)
        quality = choose_metric(lookup, exp_id, QUALITY_METRICS)
        latency = choose_metric(lookup, exp_id, LATENCY_METRICS)
        stability = choose_metric(lookup, exp_id, STABILITY_METRICS)
        compute = choose_metric(lookup, exp_id, COMPUTE_METRICS)
        if quality and latency:
            quality_points.append({"label": exp_id, "x": latency[1], "y": quality[1]})
        if stability and latency:
            stability_points.append({"label": exp_id, "x": latency[1], "y": stability[1]})
        if compute:
            compute_bars.append({"label": exp_id, "value": compute[1], "full_label": label})

    figures_dir = output_dir / "figures"
    svg_scatter(
        figures_dir / "quality_latency.svg",
        "Quality-Latency Pareto",
        quality_points,
        "Latency (mean across directions)",
        "Quality (mean across directions)",
    )
    svg_scatter(
        figures_dir / "stability_latency.svg",
        "Stability-Latency Tradeoff",
        stability_points,
        "Latency (mean across directions)",
        "Revision / Erasure Metric",
    )
    svg_bar(
        figures_dir / "compute_delay.svg",
        "Computation Cost",
        compute_bars,
        "RTF or Mean Chunk Compute",
    )


def write_trace_tables(
    output_dir: Path,
    repo_root: Path,
    manifest: dict[str, Any],
    docids: set[int],
    eval_split: str,
) -> None:
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for experiment in manifest.get("experiments", []):
        if not experiment.get("enabled", True):
            continue
        exp_id = experiment["id"]
        out_dir = experiment_output_dir(repo_root, manifest, experiment, eval_split)
        for direction in manifest.get(
            "directions", ["en-de", "en-fr", "en-nl", "en-pt", "en-ru", "en-tr"]
        ):
            metrics_log = out_dir / f"metrics_{direction}.jsonl"
            rows = read_jsonl(metrics_log)
            current_docid: int | None = None
            trace_rows = []
            for row in rows:
                if "metadata" in row:
                    current_docid = int(row.get("id", -1))
                    continue
                if current_docid not in docids or "total_audio_processed" not in row:
                    continue
                trace_rows.append(
                    {
                        "docid": current_docid,
                        "audio_s": row.get("total_audio_processed", ""),
                        "compute_s": row.get("computation_time", ""),
                        "generated": " ".join(row.get("generated_tokens", [])),
                        "deleted": " ".join(row.get("deleted_tokens", [])),
                    }
                )
            if trace_rows:
                path = traces_dir / f"{exp_id}_{direction}_trace.tsv"
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        delimiter="\t",
                        fieldnames=["docid", "audio_s", "compute_s", "generated", "deleted"],
                    )
                    writer.writeheader()
                    writer.writerows(trace_rows)


def write_markdown_report(output_dir: Path, rows: list[dict[str, str]]) -> None:
    lookup = metric_lookup(rows)
    experiments = sorted({row["experiment"] for row in rows})
    lines = [
        "# PAC Experiment Summary",
        "",
        "This report aggregates quality, latency, computation, and stability metrics from the experiment manifest.",
        "Latency values show corpus means; ± is the mean segment-level standard deviation across directions.",
        "",
        "| Experiment | Quality | Latency | Stability | Compute |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp_id in experiments:
        quality_primary = choose_metric(lookup, exp_id, QUALITY_METRICS)
        bleu = mean_metric(lookup, exp_id, "BLEU")
        quality_parts: list[str] = []
        if quality_primary:
            quality_parts.append(f"{quality_primary[0]}={quality_primary[1]:.4f}")
        if bleu is not None:
            quality_parts.append(f"BLEU={bleu:.4f}")
        quality_str = ", ".join(quality_parts)

        longlaal = mean_metric(lookup, exp_id, "LongLAAL (CA)")
        longyaal = mean_metric(lookup, exp_id, "LongYAAL (CA)")
        longlaal_stdev = mean_metric(lookup, exp_id, "LongLAAL (CA) seg_stdev")
        longyaal_stdev = mean_metric(lookup, exp_id, "LongYAAL (CA) seg_stdev")
        latency_parts: list[str] = []
        if longlaal is not None:
            part = f"LongLAAL (CA)={longlaal:.2f}"
            if longlaal_stdev is not None:
                part += f"±{longlaal_stdev:.2f}"
            latency_parts.append(part)
        if longyaal is not None:
            part = f"LongYAAL (CA)={longyaal:.2f}"
            if longyaal_stdev is not None:
                part += f"±{longyaal_stdev:.2f}"
            latency_parts.append(part)
        if not latency_parts:
            latency_primary = choose_metric(lookup, exp_id, LATENCY_METRICS)
            if latency_primary:
                latency_parts.append(f"{latency_primary[0]}={latency_primary[1]:.2f}")
        latency_str = ", ".join(latency_parts)

        stability = choose_metric(lookup, exp_id, STABILITY_METRICS)
        compute = choose_metric(lookup, exp_id, COMPUTE_METRICS)
        lines.append(
            "| {exp} | {quality} | {latency} | {stability} | {compute} |".format(
                exp=exp_id,
                quality=quality_str,
                latency=latency_str,
                stability=f"{stability[0]}={stability[1]:.4f}" if stability else "",
                compute=f"{compute[0]}={compute[1]:.4f}" if compute else "",
            )
        )
    lines.extend(
        [
            "",
            "Figures are written under `figures/`:",
            "- `quality_latency.svg`",
            "- `stability_latency.svg`",
            "- `compute_delay.svg`",
            "",
            "Trace TSV files for qualitative examples are written under `traces/`.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="pac_experiment_manifest.yaml", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Aggregate report directory.")
    parser.add_argument("--experiments", nargs="*", help="Optional experiment IDs to include.")
    parser.add_argument(
        "--set",
        dest="eval_split",
        help="Dataset split to aggregate (default: manifest mcif_set/acl_set/set).",
    )
    parser.add_argument("--trace-docids", nargs="*", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    repo_root = repo_root_from_manifest(manifest_path)
    manifest = load_manifest(manifest_path)
    eval_split = resolve_eval_split(args.eval_split, manifest)
    output_root = resolve_repo_path(repo_root, manifest.get("output_root", "output/pac_experiments"))
    report_dir = args.output_dir or (
        output_root / "report"
        if eval_split in {"eval", "test"}
        else output_root / f"report_{eval_split}"
    )
    selected_ids = set(args.experiments) if args.experiments else None

    rows, diagnostics = collect_results(repo_root, manifest, selected_ids, eval_split)
    write_summary_tsv(report_dir / "pac_experiment_summary.tsv", rows)
    (report_dir / "pac_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    write_figures(report_dir, manifest, rows)
    write_trace_tables(report_dir, repo_root, manifest, set(args.trace_docids), eval_split)
    write_markdown_report(report_dir, rows)
    print(f"Wrote PAC report: {report_dir}")


if __name__ == "__main__":
    main()
