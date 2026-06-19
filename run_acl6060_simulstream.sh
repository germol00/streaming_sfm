#!/usr/bin/env bash
# Run SimulStream offline inference (python -m simulstream.inference) with
# agent_simulstream.CascadeSpeechProcessor on ACL 60-60 eval audio.
#
# Usage:
#   ./run_acl6060_simulstream.sh
#   ./run_acl6060_simulstream.sh en-de en-pt
# Configuration: edit set_config.sh (or override env vars documented there).
#
# After inference, scores with OmniSTEval (BLEU, chrF, LongYAAL, phrase HTML report).
# Requires: pip install 'OmniSTEval[simulstream]' simulstream
#   optional COMET: pip install unbabel-comet  (or pip install 'OmniSTEval[comet]')

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_config.sh"

declare -A SOURCE_LANG=(
  [en-de]=English
  [en-fr]=English
  [en-nl]=English
  [en-pt]=English
  [en-ru]=English
  [en-tr]=English
)

declare -A TARGET_LANG=(
  [en-de]=German
  [en-fr]=French
  [en-nl]=Dutch
  [en-pt]=Portuguese
  [en-ru]=Russian
  [en-tr]=Turkish
)

if [[ $# -gt 0 ]]; then
  DIRECTIONS=("$@")
else
  DIRECTIONS=(en-de en-fr en-nl en-pt en-ru en-tr)
fi

# SimulStream's wav list loader resolves paths as: dirname(list_file) + "/" + line
# The cache FILE_ORDER lists utterance IDs without extension; wavs live in eval/full_wavs/.
LINKFARM="${OUTPUT_DIR}/eval_wav_linkfarm"
FILE_ORDER="${ACL6060_ROOT}/en-de_eval_wavs_list.txt"

if [[ ! -f "$FILE_ORDER" ]]; then
  echo "error: missing wav list / FILE_ORDER at: $FILE_ORDER" >&2
  echo "Set ACL6060_ROOT to your acl_6060 cache directory." >&2
  exit 1
fi

if [[ ! -f "$SPEECH_CFG" ]]; then
  echo "error: speech processor config not found: $SPEECH_CFG" >&2
  exit 1
fi

EVAL_DIR="$(cd "$(dirname "$(readlink -f "$FILE_ORDER" 2>/dev/null || realpath "$FILE_ORDER")")" && pwd)"
FULL_WAVS="${EVAL_DIR}/full_wavs"

if [[ ! -d "$FULL_WAVS" ]]; then
  echo "error: expected wav directory not found: $FULL_WAVS" >&2
  exit 1
fi

mkdir -p "$LINKFARM"
WAV_LIST_FILE="${LINKFARM}/wav_list.txt"
: >"$WAV_LIST_FILE"

while IFS= read -r id || [[ -n "$id" ]]; do
  [[ -z "${id// }" ]] && continue
  wav="${FULL_WAVS}/${id}.wav"
  if [[ ! -f "$wav" ]]; then
    echo "error: missing wav for id '$id': $wav" >&2
    exit 1
  fi
  ln -sf "$wav" "${LINKFARM}/${id}.wav"
  echo "${id}.wav" >>"$WAV_LIST_FILE"
done <"$FILE_ORDER"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_direction() {
  local tag="$1"
  local src_lang="$2"
  local tgt_lang="$3"
  local metrics="${OUTPUT_DIR}/metrics_${tag}.jsonl"

  echo "=== ${tag} (${src_lang} -> ${tgt_lang}) ==="
  "$PYTHON" -m simulstream.inference \
    --speech-processor-config "$SPEECH_CFG" \
    --wav-list-file "$WAV_LIST_FILE" \
    --src-lang "$src_lang" \
    --tgt-lang "$tgt_lang" \
    --metrics-log-file "$metrics"
  echo "Wrote metrics: $metrics"
}

mkdir -p "$OUTPUT_DIR"

for tag in "${DIRECTIONS[@]}"; do
  if [[ -z "${SOURCE_LANG[$tag]:-}" || -z "${TARGET_LANG[$tag]:-}" ]]; then
    echo "error: unknown direction '$tag' (supported: ${!TARGET_LANG[*]})" >&2
    exit 1
  fi
  run_direction "$tag" "${SOURCE_LANG[$tag]}" "${TARGET_LANG[$tag]}"
done

echo ""
echo "Scoring with OmniSTEval (BLEU, chrF, LongYAAL, normalized erasure, phrase report)..."
"${REPO_ROOT}/score_acl6060_metrics.sh" "${DIRECTIONS[@]}"

echo "Done. Wav list (for SimulStream): $WAV_LIST_FILE"
