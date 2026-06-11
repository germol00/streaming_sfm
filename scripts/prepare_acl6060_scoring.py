#!/usr/bin/env python3
"""Prepare ACL 60-60 scoring inputs for OmniSTEval (and legacy SimulStream tools)."""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path

import yaml


def _resolve(path: Path) -> Path:
    return path.resolve()


def _seg_counts_from_xml(xml_path: Path) -> dict[str, int]:
    text = xml_path.read_text(encoding="utf-8")
    talk_ids = re.findall(r"<talkid>([^<]+)</talkid>", text)
    parts = re.split(r"<talkid>[^<]+</talkid>", text)[1:]
    counts = [len(re.findall(r"<seg id=", part)) for part in parts]
    if len(talk_ids) != len(counts):
        raise RuntimeError(f"Could not parse segment counts from {xml_path}")
    return dict(zip(talk_ids, counts))


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _load_yaml_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, list):
        raise ValueError(f"Expected a YAML list in {path}")
    return data


def _split_ref_lines_by_talk(
    ref_path: Path,
    file_order: list[str],
    seg_counts: dict[str, int],
) -> dict[str, list[str]]:
    lines = [line.rstrip("\n") for line in ref_path.read_text(encoding="utf-8").splitlines()]
    expected = sum(seg_counts[talk] for talk in file_order)
    if len(lines) != expected:
        raise ValueError(
            f"{ref_path} has {len(lines)} lines, expected {expected} from XML segment counts"
        )
    by_talk: dict[str, list[str]] = {}
    idx = 0
    for talk in file_order:
        n = seg_counts[talk]
        by_talk[talk] = lines[idx : idx + n]
        idx += n
    return by_talk


def _load_metrics_delays_by_talk(metrics_path: Path) -> dict[str, tuple[list[float], float | None]]:
    """
    Read SimulStream metrics JSONL and return per-talk emitted-token delays.

    Returns:
      talk_stem -> (delays_ms_per_emitted_token, source_length_ms_or_None)
    """
    out: dict[str, tuple[list[float], float | None]] = {}
    if not metrics_path.is_file():
        return out

    current_id: int | None = None
    id_to_talk: dict[int, str] = {}
    delays_by_talk: dict[str, list[float]] = {}
    source_len_by_talk: dict[str, float] = {}

    with metrics_path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "id" not in row:
                continue
            rid = int(row["id"])
            current_id = rid

            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                wav_name = metadata.get("wav_name")
                if isinstance(wav_name, str):
                    talk = Path(wav_name).stem
                    id_to_talk[rid] = talk
                    delays_by_talk.setdefault(talk, [])

            talk = id_to_talk.get(rid)
            if talk is None:
                continue

            generated_tokens = row.get("generated_tokens", [])
            if generated_tokens:
                delay_ms = float(row["total_audio_processed"]) * 1000.0
                delays_by_talk[talk].extend([delay_ms] * len(generated_tokens))

            if "source_length" in row:
                source_len_by_talk[talk] = float(row["source_length"])

    for talk, delays in delays_by_talk.items():
        out[talk] = (delays, source_len_by_talk.get(talk))
    return out


def _try_existing_audio_definition(candidates: list[Path], expected_segments: int) -> Path | None:
    for candidate in candidates:
        candidate = _resolve(candidate)
        if not candidate.is_file():
            continue
        try:
            entries = _load_yaml_entries(candidate)
        except Exception:
            continue
        if len(entries) == expected_segments:
            return candidate
    return None


def build_audio_definition(
    acl_root: Path,
    output_dir: Path,
    eval_dir: Path,
    file_order: list[str],
    seg_counts: dict[str, int],
    out_yaml: Path,
) -> tuple[str, str]:
    expected_segments = sum(seg_counts[talk] for talk in file_order)
    candidates = [
        acl_root / "en-de_eval_refs.yaml",
        acl_root / "en-fr_eval_refs.yaml",
    ]
    existing = _try_existing_audio_definition(candidates, expected_segments)
    if existing is not None:
        out_yaml.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
        return f"copied gold segments from {existing}", "gold"

    metrics_path = output_dir / "metrics_en-de.jsonl"
    src_refs_path = acl_root / "en-de_eval_refs.en"
    metrics_delays = _load_metrics_delays_by_talk(metrics_path)
    ref_lines_by_talk = (
        _split_ref_lines_by_talk(src_refs_path, file_order, seg_counts)
        if src_refs_path.is_file()
        else {}
    )

    full_wavs = eval_dir / "full_wavs"
    entries: list[dict] = []
    used_metrics_offsets = False
    for talk in file_order:
        wav_name = f"{talk}.wav"
        duration_ms = _wav_duration_seconds(full_wavs / wav_name) * 1000.0
        n_segs = seg_counts[talk]
        talk_delays, source_len_ms = metrics_delays.get(talk, ([], None))
        seg_refs = ref_lines_by_talk.get(talk)

        if talk_delays and seg_refs and len(seg_refs) == n_segs:
            ref_word_counts = [max(1, len(seg.strip().split())) for seg in seg_refs]
            total_ref_words = sum(ref_word_counts)
            total_hyp_words = len(talk_delays)
            starts_ms: list[float] = [0.0]
            cum_ref_words = 0
            for i in range(1, n_segs):
                cum_ref_words += ref_word_counts[i - 1]
                frac = cum_ref_words / total_ref_words if total_ref_words > 0 else i / n_segs
                hyp_idx = int(round(frac * (total_hyp_words - 1)))
                hyp_idx = min(max(hyp_idx, 0), total_hyp_words - 1)
                starts_ms.append(talk_delays[hyp_idx])

            rec_end_ms = duration_ms if source_len_ms is None else min(duration_ms, source_len_ms)
            starts_ms = [min(max(x, 0.0), rec_end_ms) for x in starts_ms]
            for i in range(1, len(starts_ms)):
                if starts_ms[i] < starts_ms[i - 1]:
                    starts_ms[i] = starts_ms[i - 1]

            # Ensure strictly positive segment durations for OmniSTEval latency scorers.
            # Repeated emission delays can collapse adjacent boundaries to the same value.
            eps_ms = 1.0
            if n_segs > 1 and rec_end_ms <= eps_ms * (n_segs - 1):
                eps_ms = max(1e-3, rec_end_ms / (2.0 * n_segs))

            # Forward pass: enforce minimum spacing.
            for i in range(1, len(starts_ms)):
                min_allowed = starts_ms[i - 1] + eps_ms
                if starts_ms[i] < min_allowed:
                    starts_ms[i] = min_allowed

            # Backward pass: keep room before recording end.
            latest_start = rec_end_ms - eps_ms * (n_segs - 1)
            if starts_ms[0] > latest_start:
                starts_ms[0] = max(0.0, latest_start)
            for i in range(len(starts_ms) - 2, -1, -1):
                max_allowed = starts_ms[i + 1] - eps_ms
                if starts_ms[i] > max_allowed:
                    starts_ms[i] = max_allowed

            for i in range(n_segs):
                start = starts_ms[i]
                end = starts_ms[i + 1] if i + 1 < n_segs else rec_end_ms
                if end < start:
                    end = start
                entries.append(
                    {
                        "wav": wav_name,
                        "offset": round(start / 1000.0, 4),
                        "duration": round((end - start) / 1000.0, 4),
                    }
                )
            used_metrics_offsets = True
        else:
            seg_dur_ms = duration_ms / n_segs
            offset_ms = 0.0
            for _ in range(n_segs):
                entries.append(
                    {
                        "wav": wav_name,
                        "offset": round(offset_ms / 1000.0, 4),
                        "duration": round(seg_dur_ms / 1000.0, 4),
                    }
                )
                offset_ms += seg_dur_ms

    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, sort_keys=False)
    if used_metrics_offsets:
        return (
            f"built metrics-informed segment timings ({expected_segments} segments) "
            f"using {metrics_path.name}"
        ), "metrics_inferred"
    return (
        f"built proportional segment timings ({expected_segments} segments) "
        "(set ACL6060_ROOT/.../en-de_eval_refs.yaml for gold timings)"
    ), "proportional"


def split_reference_file(
    ref_path: Path,
    file_order: list[str],
    seg_counts: dict[str, int],
    out_dir: Path,
    suffix: str,
) -> list[Path]:
    lines = [line.rstrip("\n") for line in ref_path.read_text(encoding="utf-8").splitlines()]
    expected = sum(seg_counts[talk] for talk in file_order)
    if len(lines) != expected:
        raise ValueError(
            f"{ref_path} has {len(lines)} lines, expected {expected} from XML segment counts"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files: list[Path] = []
    idx = 0
    for talk in file_order:
        n = seg_counts[talk]
        chunk = lines[idx : idx + n]
        idx += n
        out_file = out_dir / f"{talk}{suffix}"
        out_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        out_files.append(out_file)
    return out_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acl-root",
        type=Path,
        default=Path.home() / ".cache/simuleval/acl_6060",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    acl_root = _resolve(args.acl_root)
    output_dir = _resolve(args.output_dir)
    scoring_dir = output_dir / "scoring_data"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    file_order_path = acl_root / "en-de_eval_wavs_list.txt"
    file_order_path = _resolve(file_order_path)
    eval_dir = file_order_path.parent
    file_order = [
        line.strip()
        for line in file_order_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    xml_path = eval_dir / "text/xml/ACL.6060.eval.en-xx.de.xml"
    seg_counts = _seg_counts_from_xml(xml_path)

    audio_yaml = scoring_dir / "audio_definition.yaml"
    note, timing_source = build_audio_definition(
        acl_root, output_dir, eval_dir, file_order, seg_counts, audio_yaml
    )
    print(note)
    print(f"Wrote {audio_yaml}")
    timing_meta = scoring_dir / "audio_definition.meta.json"
    timing_meta.write_text(
        json.dumps({"timing_source": timing_source}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {timing_meta}")

    directions = {
        "en-de": (acl_root / "en-de_eval_refs.de", acl_root / "en-de_eval_refs.en"),
        "en-fr": (acl_root / "en-fr_eval_refs.fr", acl_root / "en-fr_eval_refs.en"),
        "en-pt": (acl_root / "en-pt_eval_refs.pt", acl_root / "en-pt_eval_refs.en"),
    }
    for tag, (ref_path, src_path) in directions.items():
        ref_path = _resolve(ref_path)
        src_path = _resolve(src_path)
        ref_out = scoring_dir / f"refs_{tag}"
        src_out = scoring_dir / f"transcripts_{tag}"
        for out_dir in (ref_out, src_out):
            if out_dir.exists():
                for old in out_dir.glob("*.txt"):
                    old.unlink()
        # File stems must match wav stems (e.g. 2022.acl-long.410) for SimulStream LogReader.
        ref_files = split_reference_file(
            ref_path,
            file_order,
            seg_counts,
            ref_out,
            ".txt",
        )
        src_files = split_reference_file(
            src_path,
            file_order,
            seg_counts,
            src_out,
            ".txt",
        )
        merged_ref = scoring_dir / f"refs_{tag}_merged.txt"
        merged_src = scoring_dir / f"sources_{tag}_merged.txt"
        merged_ref.write_text(ref_path.read_text(encoding="utf-8"), encoding="utf-8")
        merged_src.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(
            f"{tag}: {len(ref_files)} per-talk refs, merged refs/sources for OmniSTEval "
            f"({merged_ref.name}, {merged_src.name})"
        )


if __name__ == "__main__":
    main()
