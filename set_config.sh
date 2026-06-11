#!/usr/bin/env bash
# Shared configuration for ACL 60-60 SimulStream inference and scoring.
#
# Sourced by run_acl6060_simulstream.sh and score_acl6060_metrics.sh.
# Edit the defaults below, or override any variable when invoking the scripts:
#   SPEECH_CFG=/path/to/other.yaml OUTPUT_DIR=/tmp/my_run ./run_acl6060_simulstream.sh
#
# Main knobs:
#   SPEECH_CFG     speech_processor YAML (drives model / pipeline settings)
#   OUTPUT_DIR     metrics logs, scoring data, OmniSTEval outputs (default: output/<config-basename>)
#   ACL6060_ROOT   ACL 60-60 dataset cache
#   PYTHON         Python interpreter
#   BLEU_TOKENIZER SacreBLEU tokenizer for OmniSTEval (default: intl)
#   HTML_MAX_SEGS  Max segments in phrase HTML report (0 = all)
#   SKIP_COMET=1   Skip COMET scoring

_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$_CONFIG_DIR"
REPO_ROOT="$SCRIPT_DIR"

ACL6060_ROOT="${ACL6060_ROOT:-${HOME}/.cache/simuleval/acl_6060}"

# Default speech processor config (change this line to switch experiments).
SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor_qwen35_9b_bnb4_spec.yaml}"
#SPEECH_CFG="${SPEECH_CFG:-${REPO_ROOT}/speech_processor.yaml}"

speechp_name="$(basename "${SPEECH_CFG}" .yaml)"
#OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/simulstream_acl6060}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/${speechp_name}}"

PYTHON="${PYTHON:-python3}"
BLEU_TOKENIZER="${BLEU_TOKENIZER:-13a}"
HTML_MAX_SEGS="${HTML_MAX_SEGS:-0}"

SCORING_DIR="${OUTPUT_DIR}/scoring_data"
RESULTS_TSV="${OUTPUT_DIR}/scores.tsv"
PREDICTS="${OUTPUT_DIR}/preds.txt"
