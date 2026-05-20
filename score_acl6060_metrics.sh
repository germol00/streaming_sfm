#!/usr/bin/env bash
# Score SimulStream metrics logs (BLEU, COMET, LAAL) for ACL 60-60 en-de / en-fr runs.
#
# Usage:
#   ./score_acl6060_metrics.sh
#   ./score_acl6060_metrics.sh en-de
#
# Env:
#   OUTPUT_DIR   Directory with metrics_en-*.jsonl (default: output/simulstream_acl6060)
#   ACL6060_ROOT Dataset cache (default: ~/.cache/simuleval/acl_6060)
#   SPEECH_CFG   speech_processor.yaml for detokenizer settings
#   PYTHON       Python interpreter
#   SKIP_COMET=1 Skip COMET (no GPU / model download)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/simulstream_acl6060}"
ACL6060_ROOT="${ACL6060_ROOT:-${HOME}/.cache/simuleval/acl_6060}"
SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor.yaml}"
PYTHON="${PYTHON:-python3}"
SCORING_DIR="${OUTPUT_DIR}/scoring_data"
RESULTS_TSV="${OUTPUT_DIR}/scores.tsv"

if [[ $# -gt 0 ]]; then
  DIRECTIONS=("$@")
else
  DIRECTIONS=(en-de en-fr)
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! "$PYTHON" -c "import mweralign" 2>/dev/null; then
  for candidate in \
    "${HOME}/miniconda3/envs/evaluation/bin/python" \
    "${HOME}/miniconda3/envs/iwslt/bin/python" \
    "${HOME}/miniconda3/envs/simuleval/bin/python"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "import mweralign" 2>/dev/null; then
      PYTHON="$candidate"
      echo "Using PYTHON=$PYTHON (has mweralign)"
      break
    fi
  done
fi
if ! "$PYTHON" -c "import mweralign" 2>/dev/null; then
  echo "error: mweralign is required for SimulStream BLEU/COMET/LAAL scoring." >&2
  echo "Install with: pip install mweralign   # or: pip install 'simulstream[eval]'" >&2
  exit 1
fi

echo "Preparing per-talk references and audio definition..."
"$PYTHON" "${REPO_ROOT}/scripts/prepare_acl6060_scoring.py" \
  --acl-root "$ACL6060_ROOT" \
  --output-dir "$OUTPUT_DIR"

AUDIO_DEF="${SCORING_DIR}/audio_definition.yaml"
if [[ ! -f "$AUDIO_DEF" ]]; then
  echo "error: missing $AUDIO_DEF" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
: >"$RESULTS_TSV"
printf "direction\tmetric\tvalue\tdetails\n" >>"$RESULTS_TSV"

score_direction() {
  local tag="$1"
  local metrics_log="${OUTPUT_DIR}/metrics_${tag}.jsonl"
  local refs_dir="${SCORING_DIR}/refs_${tag}"
  local src_dir="${SCORING_DIR}/transcripts_${tag}"

  if [[ ! -f "$metrics_log" ]]; then
    echo "error: metrics log not found: $metrics_log (run inference first)" >&2
    return 1
  fi

  mapfile -t REF_FILES < <(find "$refs_dir" -name '*.txt' | sort)
  mapfile -t SRC_FILES < <(find "$src_dir" -name '*.txt' | sort)

  if [[ ${#REF_FILES[@]} -eq 0 ]]; then
    echo "error: no reference files in $refs_dir" >&2
    return 1
  fi

  echo ""
  echo "========== ${tag} =========="

  echo "--- SacreBLEU ---"
  local bleu_out
  bleu_out=$("$PYTHON" -m simulstream.metrics.score_quality \
    --eval-config "$SPEECH_CFG" \
    --log-file "$metrics_log" \
    --scorer sacrebleu \
    --tokenizer intl \
    --references "${REF_FILES[@]}" 2>&1 | tee /dev/stderr)
  local bleu
  bleu=$(echo "$bleu_out" | awk -F': ' '/sacrebleu score:/ {print $2; exit}')
  printf "%s\tBLEU\t%s\tintl tokenizer\n" "$tag" "$bleu" >>"$RESULTS_TSV"

  if [[ "${SKIP_COMET:-0}" != "1" ]]; then
  if "$PYTHON" -c "import comet" 2>/dev/null; then
    echo "--- COMET ---"
    local comet_out comet
    comet_out=$("$PYTHON" -m simulstream.metrics.score_quality \
      --eval-config "$SPEECH_CFG" \
      --log-file "$metrics_log" \
      --scorer comet \
      --references "${REF_FILES[@]}" \
      --transcripts "${SRC_FILES[@]}" 2>&1 | tee /dev/stderr)
    comet=$(echo "$comet_out" | awk -F': ' '/comet score:/ {print $2; exit}')
    printf "%s\tCOMET\t%s\twmt22-comet-da\n" "$tag" "$comet" >>"$RESULTS_TSV"
  else
    echo "Skipping COMET (install with: pip install unbabel-comet)" >&2
  fi
  else
    echo "Skipping COMET (SKIP_COMET=1)" >&2
  fi

  echo "--- LAAL (stream_laal) ---"
  # LAAL uses one line-per-segment reference file aligned with audio_definition (all talks).
  local merged_ref="${SCORING_DIR}/refs_${tag}_merged.txt"
  cat "${REF_FILES[@]}" >"$merged_ref"

  local laal_out laal_ideal laal_ca
  laal_out=$("$PYTHON" -m simulstream.metrics.score_latency \
    --eval-config "$SPEECH_CFG" \
    --log-file "$metrics_log" \
    --scorer stream_laal \
    --audio-definition "$AUDIO_DEF" \
    --reference "$merged_ref" \
    --latency-unit word 2>&1 | tee /dev/stderr)
  laal_ideal=$(echo "$laal_out" | sed -n 's/.*ideal_latency=\([0-9.eE+-]*\).*/\1/p')
  laal_ca=$(echo "$laal_out" | sed -n 's/.*computational_aware_latency=\([0-9.eE+-]*\).*/\1/p')
  printf "%s\tLAAL\t%s\tideal (seconds)\n" "$tag" "$laal_ideal" >>"$RESULTS_TSV"
  printf "%s\tLAAL-CA\t%s\tcomputation-aware (seconds)\n" "$tag" "$laal_ca" >>"$RESULTS_TSV"
}

for tag in "${DIRECTIONS[@]}"; do
  score_direction "$tag"
done

echo ""
echo "Wrote summary: $RESULTS_TSV"
column -t -s $'\t' "$RESULTS_TSV" 2>/dev/null || cat "$RESULTS_TSV"
