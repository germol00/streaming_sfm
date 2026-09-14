#!/usr/bin/env python3
"""Plot EAC window-size sweep: latency vs quality across chunk sizes.

Reads aggregated metrics from the EAC experiment report (README.md table in
output/pac_experiments_wsize/report/) and maps window sizes from
pac_experiment_manifest_wsize.yaml. Falls back to per-experiment scores.tsv
when the report is missing.

Usage (from repo root):
    python3 plots/pac_window_size.py
    python3 plots/pac_window_size.py --manifest pac_experiment_manifest_wsize.yaml
    python3 plots/pac_window_size.py --report-dir output/pac_experiments_wsize/report
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


LATENCY_AWARE = "LongYAAL (CA)"
LATENCY_UNAWARE = "LongLAAL (CA)"
QUALITY_METRIC = "COMET"
COMET_SCALE = 100.0
MS_PER_S = 1000.0
DEFAULT_L_MAX = 20
L_MAX_COLORS = {20: "tab:green", 10: "tab:red"}
L_MAX_MARKERS = {20: "s", 10: "o"}
QUALITY_LINEWIDTH = 2.5
LATENCY_LINEWIDTH = 1.25
LEGEND_INSET = 0.04


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Manifest must be a YAML mapping: {path}")
    return manifest


def resolve_repo_path(repo_root_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root_path / path


def experiment_output_dir(
    repo_root_path: Path,
    manifest: dict[str, Any],
    experiment: dict[str, Any],
    eval_split: str,
) -> Path:
    output_dir = experiment.get("output_dir")
    if output_dir:
        base = resolve_repo_path(repo_root_path, output_dir)
    else:
        output_root = resolve_repo_path(
            repo_root_path, manifest.get("output_root", "output/pac_experiments")
        )
        base = output_root / experiment["id"]
    if eval_split not in {"eval", "test"}:
        base = base / eval_split
    return base


def chunk_size_from_id(experiment_id: str) -> float | None:
    match = re.search(r"lacp(\d+)", experiment_id)
    if match:
        return int(match.group(1)) / 100.0
    return None


def max_len_from_id(experiment_id: str) -> int:
    if re.search(r"_10$", experiment_id):
        return 10
    return DEFAULT_L_MAX


def max_len_from_experiment(experiment: dict[str, Any]) -> int:
    overrides = experiment.get("overrides", {})
    if "sfm_left_context_secs" in overrides:
        return int(float(overrides["sfm_left_context_secs"]))
    return max_len_from_id(experiment.get("id", ""))


def chunk_size_from_experiment(experiment: dict[str, Any]) -> float | None:
    overrides = experiment.get("overrides", {})
    for key in ("speech_chunk_size", "sfm_chunk_secs"):
        if key in overrides:
            return float(overrides[key])
    return chunk_size_from_id(experiment.get("id", ""))


def experiment_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        experiment["id"]: experiment
        for experiment in manifest.get("experiments", [])
        if experiment.get("enabled", True)
    }


def default_report_dir(
    repo_root_path: Path,
    manifest: dict[str, Any],
    eval_split: str,
) -> Path:
    output_root = resolve_repo_path(
        repo_root_path, manifest.get("output_root", "output/pac_experiments")
    )
    if eval_split in {"eval", "test"}:
        return output_root / "report"
    return output_root / f"report_{eval_split}"


def parse_metric_cell(cell: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for part in cell.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        try:
            metrics[name.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def parse_report_readme(path: Path) -> list[dict[str, Any]]:
    """Parse the aggregate README.md table written by pac_experiment_report.py."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("| pac_"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue

        experiment_id = parts[1]
        quality = parse_metric_cell(parts[2])
        latency = parse_metric_cell(parts[3])
        comet = quality.get(QUALITY_METRIC)
        aware_ms = latency.get(LATENCY_AWARE)
        unaware_ms = latency.get(LATENCY_UNAWARE)
        if comet is None or aware_ms is None or unaware_ms is None:
            print(
                f"Warning: incomplete metrics in report row for {experiment_id}",
                file=sys.stderr,
            )
            continue

        records.append(
            {
                "experiment": experiment_id,
                "Aware_Latency": aware_ms / MS_PER_S,
                "Unaware_Latency": unaware_ms / MS_PER_S,
                QUALITY_METRIC: comet,
            }
        )
    return records


def parse_report_summary(
    path: Path,
    directions: list[str],
) -> list[dict[str, Any]]:
    """Parse pac_experiment_summary.tsv when README.md is unavailable."""
    rows = read_scores(path)
    if not rows:
        return []

    by_experiment: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment"], []).append(row)

    records: list[dict[str, Any]] = []
    for experiment_id, exp_rows in by_experiment.items():
        aware_ms = mean_metric(exp_rows, directions, LATENCY_AWARE)
        unaware_ms = mean_metric(exp_rows, directions, LATENCY_UNAWARE)
        comet = mean_metric(exp_rows, directions, QUALITY_METRIC)
        if aware_ms is None or unaware_ms is None or comet is None:
            print(
                f"Warning: incomplete metrics in summary for {experiment_id}",
                file=sys.stderr,
            )
            continue
        records.append(
            {
                "experiment": experiment_id,
                "Aware_Latency": aware_ms / MS_PER_S,
                "Unaware_Latency": unaware_ms / MS_PER_S,
                QUALITY_METRIC: comet,
            }
        )
    return records


def enrich_with_manifest(
    records: list[dict[str, Any]],
    experiments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        experiment_id = record["experiment"]
        experiment = experiments.get(experiment_id, {"id": experiment_id})
        chunk_size = chunk_size_from_experiment(experiment)
        if chunk_size is None:
            print(
                f"Warning: could not infer chunk size for {experiment_id}",
                file=sys.stderr,
            )
            continue
        enriched.append(
            {
                **record,
                "label": experiment.get("label", experiment_id),
                "chunk_size": chunk_size,
                "l_max": max_len_from_experiment(experiment),
            }
        )
    return enriched


def read_scores(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean_metric(rows: list[dict[str, str]], directions: list[str], metric: str) -> float | None:
    values: list[float] = []
    direction_set = set(directions)
    for row in rows:
        if row.get("metric") != metric:
            continue
        if direction_set and row.get("direction") not in direction_set:
            continue
        try:
            values.append(float(row["value"]))
        except (KeyError, ValueError):
            continue
    if not values:
        return None
    return statistics.fmean(values)


def collect_window_size_data_from_scores(
    repo_root_path: Path,
    manifest: dict[str, Any],
    eval_split: str,
) -> list[dict[str, Any]]:
    directions = manifest.get(
        "directions", ["en-de", "en-fr", "en-nl", "en-pt", "en-ru", "en-tr"]
    )
    records: list[dict[str, Any]] = []

    for experiment in manifest.get("experiments", []):
        if not experiment.get("enabled", True):
            continue
        chunk_size = chunk_size_from_experiment(experiment)
        if chunk_size is None:
            print(f"Warning: could not infer chunk size for {experiment['id']}", file=sys.stderr)
            continue

        out_dir = experiment_output_dir(repo_root_path, manifest, experiment, eval_split)
        scores_path = out_dir / "scores.tsv"
        rows = read_scores(scores_path)
        if not rows:
            print(f"Warning: no scores found at {scores_path}", file=sys.stderr)
            continue

        aware_ms = mean_metric(rows, directions, LATENCY_AWARE)
        unaware_ms = mean_metric(rows, directions, LATENCY_UNAWARE)
        comet = mean_metric(rows, directions, QUALITY_METRIC)
        if aware_ms is None or unaware_ms is None or comet is None:
            print(
                f"Warning: incomplete metrics for {experiment['id']} ({scores_path})",
                file=sys.stderr,
            )
            continue

        records.append(
            {
                "experiment": experiment["id"],
                "label": experiment.get("label", experiment["id"]),
                "chunk_size": chunk_size,
                "l_max": max_len_from_experiment(experiment),
                "Aware_Latency": aware_ms / MS_PER_S,
                "Unaware_Latency": unaware_ms / MS_PER_S,
                QUALITY_METRIC: comet,
            }
        )
    return records


def collect_window_size_data_from_report(
    report_dir: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]] | None:
    readme_path = report_dir / "README.md"
    if readme_path.exists():
        print(f"Reading aggregate metrics from {readme_path}", file=sys.stderr)
        records = parse_report_readme(readme_path)
    else:
        summary_path = report_dir / "pac_experiment_summary.tsv"
        if not summary_path.exists():
            return None
        print(f"Reading aggregate metrics from {summary_path}", file=sys.stderr)
        directions = manifest.get(
            "directions", ["en-de", "en-fr", "en-nl", "en-pt", "en-ru", "en-tr"]
        )
        records = parse_report_summary(summary_path, directions)

    if not records:
        return None
    return enrich_with_manifest(records, experiment_lookup(manifest))


def collect_window_size_data(
    repo_root_path: Path,
    manifest: dict[str, Any],
    eval_split: str,
    report_dir: Path | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    columns = [
        "experiment",
        "label",
        "chunk_size",
        "l_max",
        "Aware_Latency",
        "Unaware_Latency",
        QUALITY_METRIC,
    ]
    records: list[dict[str, Any]] | None = None

    if source in {"auto", "report"}:
        resolved_report_dir = report_dir or default_report_dir(
            repo_root_path, manifest, eval_split
        )
        records = collect_window_size_data_from_report(resolved_report_dir, manifest)
        if records is None and source == "report":
            raise SystemExit(f"No aggregate report found under {resolved_report_dir}")

    if records is None and source in {"auto", "scores"}:
        if source == "auto":
            print("Falling back to per-experiment scores.tsv files.", file=sys.stderr)
        records = collect_window_size_data_from_scores(repo_root_path, manifest, eval_split)

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(["l_max", "chunk_size"])


def padded_limits(values: pd.Series, pad_frac: float = 0.08) -> tuple[float, float]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        delta = max(abs(vmin) * 0.05, 0.05)
        return vmin - delta, vmax + delta
    pad = (vmax - vmin) * pad_frac
    return vmin - pad, vmax + pad


def plot_window_size(df: pd.DataFrame, output_path: Path, series_label: str = "EAC") -> None:
    fig, ax1 = plt.subplots(figsize=(9, 9))
    ax2 = ax1.twinx()
    ax1.set_box_aspect(1)

    legend_handles = []
    legend_labels = []

    if not df.empty:
        all_aware: list[float] = []
        all_comet: list[float] = []

        for l_max in sorted(df["l_max"].unique(), reverse=True):
            subset = df[df["l_max"] == l_max].sort_values("chunk_size")
            color = L_MAX_COLORS.get(int(l_max), "tab:blue")
            marker = L_MAX_MARKERS.get(int(l_max), "s")
            comet_scaled = subset["COMET"] * COMET_SCALE
            all_aware.extend(subset["Aware_Latency"].tolist())
            all_comet.extend(comet_scaled.tolist())

            h_aware, = ax1.plot(
                subset["chunk_size"],
                subset["Aware_Latency"],
                color=color,
                linestyle="--",
                linewidth=LATENCY_LINEWIDTH,
                marker=marker,
            )
            h_comet, = ax2.plot(
                subset["chunk_size"],
                comet_scaled,
                color=color,
                linestyle="-",
                linewidth=QUALITY_LINEWIDTH,
                marker="x",
            )
            legend_handles.extend([h_aware, h_comet])
            legend_labels.extend(
                [
                    rf"{series_label} $L_{{\max}}={int(l_max)}$ YAAL",
                    rf"{series_label} $L_{{\max}}={int(l_max)}$ {QUALITY_METRIC}",
                ]
            )

        ax1.set_ylim(*padded_limits(pd.Series(all_aware)))
        ax2.set_ylim(*padded_limits(pd.Series(all_comet)))

    ax1.text(
        0.02,
        0.97,
        "YAAL (s)",
        transform=ax1.transAxes,
        fontweight="bold",
        fontsize=15,
    )
    ax1.text(
        0.88,
        0.97,
        QUALITY_METRIC,
        transform=ax1.transAxes,
        fontweight="bold",
        fontsize=15,
    )
    ax1.text(
        0.02,
        0.02,
        r"$\mathbf{L_c}$ (s)",
        transform=ax1.transAxes,
        fontweight="bold",
        fontsize=15,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)

    if legend_handles:
        ax1.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            bbox_to_anchor=(1.0 - LEGEND_INSET, LEGEND_INSET),
            fontsize="x-large",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Wrote {output_path}")


def resolve_eval_split(cli_value: str | None, manifest: dict[str, Any]) -> str:
    return cli_value or manifest.get("acl_set") or manifest.get("set", "eval")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "pac_experiment_manifest_wsize.yaml",
        help="EAC experiment manifest with window-size sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "plots" / "pac_window_size.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Aggregate report directory containing README.md (default: <output_root>/report).",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "report", "scores"),
        default="auto",
        help="Metric source: aggregate report (default), or per-experiment scores.tsv.",
    )
    parser.add_argument(
        "--set",
        dest="eval_split",
        help="Dataset split subdirectory (default: manifest acl_set/set or eval).",
    )
    parser.add_argument(
        "--series-label",
        default="EAC",
        help="Legend prefix for the plotted condition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    eval_split = resolve_eval_split(args.eval_split, manifest)

    report_dir = (
        args.report_dir.resolve()
        if args.report_dir
        else default_report_dir(root, manifest, eval_split)
    )
    df = collect_window_size_data(
        root,
        manifest,
        eval_split,
        report_dir=report_dir,
        source=args.source,
    )
    if df.empty:
        print(
            "No experiment metrics found yet; writing an empty styled figure.",
            file=sys.stderr,
        )
    else:
        l_max_values = sorted(df["l_max"].unique())
        print(
            f"Loaded {len(df)} window-size points "
            f"for L_max={', '.join(str(int(v)) for v in l_max_values)}"
        )

    plot_window_size(df, args.output.resolve(), series_label=args.series_label)


if __name__ == "__main__":
    main()
