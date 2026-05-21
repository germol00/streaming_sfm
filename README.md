# streaming_sfm

Streaming speech foundation model (SFM) ASR with SimulStream cascade MT evaluation on ACL 60-60.

## ACL 60-60 evaluation (OmniSTEval)

Inference writes SimulStream metrics logs; scoring uses [OmniSTEval](https://github.com/pe-trik/OmniSTEval) longform resegmentation (same toolchain as recent IWSLT shared tasks): corpus BLEU/chrF/LongYAAL plus per-phrase outputs.

```bash
pip install -r requirements-eval.txt
# optional: pip install unbabel-comet

./run_acl6060_simulstream.sh          # inference + scoring
./score_acl6060_metrics.sh en-de     # score existing logs only
```

**Outputs** (under `OUTPUT_DIR`, default `output/simulstream_acl6060`):

| Path | Description |
|------|-------------|
| `omnisteval_<dir>/evaluation_report.txt` | Full OmniSTEval report |
| `omnisteval_<dir>/scores.tsv` | Corpus metrics |
| `omnisteval_<dir>/instances.resegmented.jsonl` | One JSON object per reference segment (prediction + reference + delays) |
| `omnisteval_<dir>/phrase_report.html` | Word-level diff per segment (open in a browser) |
| `scores.tsv` | Combined summary across directions |

**Environment**

- `ACL6060_ROOT` — ACL 60-60 cache (default `~/.cache/simuleval/acl_6060`)
- `SPEECH_CFG` — `speech_processor.yaml` (must match inference; used as SimulStream eval config)
- `BLEU_TOKENIZER` — SacreBLEU tokenizer (default `intl`)
- `SKIP_COMET=1` — skip COMET
- `HTML_MAX_SEGS=N` — cap segments in the HTML report (0 = all)

Gold segment timings are used when `en-de_eval_refs.yaml` / `en-fr_eval_refs.yaml` exist in the ACL cache; otherwise proportional splits are built from wav duration.
