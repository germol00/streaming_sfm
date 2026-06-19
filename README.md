# streaming_sfm

Streaming speech foundation model (SFM) ASR with SimulStream cascade MT evaluation on ACL 60-60.

## ACL 60-60 evaluation (OmniSTEval)

Inference writes SimulStream metrics logs; scoring uses [OmniSTEval](https://github.com/pe-trik/OmniSTEval) longform resegmentation (same toolchain as recent IWSLT shared tasks): corpus BLEU/chrF/LongYAAL plus per-phrase outputs. Flickering is measured with **normalized erasure** from `simulstream.metrics.stats` (deleted tokens / final tokens; [IWSLT 2020 re-translation paper](https://aclanthology.org/2020.iwslt-1.27/)).

```bash
pip install -r requirements-eval.txt
# optional: pip install unbabel-comet

./run_acl6060_simulstream.sh          # inference + scoring (eval set)
./run_acl6060_simulstream.sh dev      # dev set
./score_acl6060_metrics.sh en-de     # score existing logs only
./score_acl6060_metrics.sh dev en-de # score dev logs only
```

**Outputs** (under `OUTPUT_DIR`, default `output/simulstream_acl6060`):

| Path | Description |
|------|-------------|
| `omnisteval_<dir>/evaluation_report.txt` | Full OmniSTEval report |
| `omnisteval_<dir>/scores.tsv` | Corpus metrics |
| `omnisteval_<dir>/instances.resegmented.jsonl` | One JSON object per reference segment (prediction + reference + delays) |
| `omnisteval_<dir>/phrase_report.html` | Word-level diff per segment (open in a browser) |
| `scores.tsv` | Combined summary across directions |
| `omnisteval_<dir>/stats.json` | Normalized erasure and real-time factor from the metrics log |

**Environment**

- `ACL6060_ROOT` — ACL 60-60 cache (default `~/.cache/simuleval/acl_6060`)
- `ACL6060_SET` — `eval` (default) or `dev`
- `SPEECH_CFG` — `speech_processor.yaml` (must match inference; used as SimulStream eval config). Presets: `speech_processor_qwen35_9b.yaml` (fp16 9B), `speech_processor_qwen35_9b_bnb4.yaml` / `speech_processor_qwen35_27b_bnb4.yaml` (4-bit via vLLM bitsandbytes; `pip install 'bitsandbytes>=0.49.2'`).
- `BLEU_TOKENIZER` — SacreBLEU tokenizer (default `intl`)
- `SKIP_COMET=1` — skip COMET
- `LATENCY_UNIT` — `word` or `char` for normalized erasure (default: `latency_unit` from `SPEECH_CFG`)
- `HTML_MAX_SEGS=N` — cap segments in the HTML report (0 = all)

Gold segment timings are used when `en-de_<set>_refs.yaml` / `en-fr_<set>_refs.yaml` exist in the ACL cache; otherwise proportional splits are built from wav duration.

## Provisional ASR Context experiments

`pac_experiment_manifest.yaml` defines the paper-oriented experiment suite for Provisional ASR Context (PAC) decoding: committed-ASR baselines, PAC runs, an aggressive low-latency baseline, chunk-size sweeps, and PAC ablations.

```bash
./run_pac_experiments.sh --generate-only
./run_pac_experiments.sh --experiments committed_asr_lacp064 pac_lacp064 --directions en-de
./run_pac_experiments.sh --set dev --experiments committed_asr_lacp064 --directions en-de
./run_pac_experiments.sh --score-only
python3 scripts/pac_experiment_report.py --manifest pac_experiment_manifest.yaml
```

Generated configs are written to `output/pac_experiments/configs/`. Each experiment writes its normal SimulStream/OmniSTEval outputs under `output/pac_experiments/<experiment_id>/` (eval) or `output/pac_experiments/<experiment_id>/dev/`, while the aggregate paper summary is written to `output/pac_experiments/report/` (or `report_dev/` for dev runs).

The aggregate report includes `pac_experiment_summary.tsv`, `pac_diagnostics.json`, SVG figures for quality-latency, stability-latency, and compute cost, plus trace TSVs for qualitative streaming examples.
