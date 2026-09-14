# PAC Experiment Summary

This report aggregates quality, latency, computation, and stability metrics from the experiment manifest.

| Experiment | Quality | Latency | Stability | Compute |
|---|---:|---:|---:|---:|
| pac_lacp064 | COMET=0.6642, BLEU=32.4423 | LongLAAL (CA)=2371.34, LongYAAL (CA)=2206.02 | normalized_erasure=2.2812 | real_time_factor=0.9927 |
| pac_lacp064_10 | COMET=0.6585, BLEU=32.7794 | LongLAAL (CA)=2452.40, LongYAAL (CA)=2293.11 | normalized_erasure=2.2236 | real_time_factor=0.9997 |
| pac_lacp072 | COMET=0.6971, BLEU=35.5892 | LongLAAL (CA)=2596.69, LongYAAL (CA)=2405.29 | normalized_erasure=1.9286 | real_time_factor=0.9257 |
| pac_lacp072_10 | COMET=0.6955, BLEU=34.4759 | LongLAAL (CA)=2551.89, LongYAAL (CA)=2366.82 | normalized_erasure=2.4818 | real_time_factor=0.9197 |
| pac_lacp080 | COMET=0.6970, BLEU=35.3680 | LongLAAL (CA)=2761.16, LongYAAL (CA)=2556.67 | normalized_erasure=2.0833 | real_time_factor=0.8669 |
| pac_lacp080_10 | COMET=0.7014, BLEU=35.8307 | LongLAAL (CA)=2696.61, LongYAAL (CA)=2496.21 | normalized_erasure=2.0624 | real_time_factor=0.8557 |
| pac_lacp088 | COMET=0.7290, BLEU=37.2606 | LongLAAL (CA)=2976.04, LongYAAL (CA)=2675.00 | normalized_erasure=2.0316 | real_time_factor=0.8206 |
| pac_lacp088_10 | COMET=0.7064, BLEU=36.7516 | LongLAAL (CA)=2901.99, LongYAAL (CA)=2654.62 | normalized_erasure=2.1159 | real_time_factor=0.8141 |
| pac_lacp096 | COMET=0.7205, BLEU=37.2272 | LongLAAL (CA)=3062.81, LongYAAL (CA)=2798.46 | normalized_erasure=2.0402 | real_time_factor=0.7791 |
| pac_lacp096_10 | COMET=0.7307, BLEU=37.0306 | LongLAAL (CA)=3035.93, LongYAAL (CA)=2756.28 | normalized_erasure=2.0580 | real_time_factor=0.7719 |
| pac_lacp104 | COMET=0.7330, BLEU=37.5966 | LongLAAL (CA)=3321.91, LongYAAL (CA)=2942.14 | normalized_erasure=1.8528 | real_time_factor=0.7479 |
| pac_lacp104_10 | COMET=0.7351, BLEU=37.9306 | LongLAAL (CA)=3343.57, LongYAAL (CA)=2948.60 | normalized_erasure=2.0668 | real_time_factor=0.7604 |
| pac_lacp112 | COMET=0.7566, BLEU=38.3726 | LongLAAL (CA)=3402.13, LongYAAL (CA)=3057.20 | normalized_erasure=1.8206 | real_time_factor=0.7073 |
| pac_lacp112_10 | COMET=0.7481, BLEU=38.2716 | LongLAAL (CA)=3553.40, LongYAAL (CA)=3087.19 | normalized_erasure=2.0002 | real_time_factor=0.7320 |
| pac_lacp120 | COMET=0.7425, BLEU=38.1264 | LongLAAL (CA)=3651.55, LongYAAL (CA)=3222.60 | normalized_erasure=2.0423 | real_time_factor=0.6763 |
| pac_lacp120_10 | COMET=0.7489, BLEU=38.4319 | LongLAAL (CA)=3717.08, LongYAAL (CA)=3251.51 | normalized_erasure=2.2529 | real_time_factor=0.6870 |
| pac_lacp128 | COMET=0.7479, BLEU=37.5762 | LongLAAL (CA)=3721.79, LongYAAL (CA)=3342.92 | normalized_erasure=1.7940 | real_time_factor=0.6496 |
| pac_lacp128_10 | COMET=0.7606, BLEU=38.6861 | LongLAAL (CA)=3815.06, LongYAAL (CA)=3312.01 | normalized_erasure=1.9600 | real_time_factor=0.6605 |
| pac_lacp136 | COMET=0.7681, BLEU=38.9840 | LongLAAL (CA)=3930.39, LongYAAL (CA)=3455.92 | normalized_erasure=1.7821 | real_time_factor=0.6351 |
| pac_lacp136_10 | COMET=0.7656, BLEU=39.1195 | LongLAAL (CA)=4039.38, LongYAAL (CA)=3440.68 | normalized_erasure=1.8919 | real_time_factor=0.6412 |
| pac_lacp144 | COMET=0.7605, BLEU=38.1207 | LongLAAL (CA)=4165.71, LongYAAL (CA)=3609.17 | normalized_erasure=1.9127 | real_time_factor=0.6252 |
| pac_lacp144_10 | COMET=0.7643, BLEU=39.0218 | LongLAAL (CA)=4263.13, LongYAAL (CA)=3591.56 | normalized_erasure=1.9824 | real_time_factor=0.6280 |

Figures are written under `figures/`:
- `quality_latency.svg`
- `stability_latency.svg`
- `compute_delay.svg`

Trace TSV files for qualitative examples are written under `traces/`.
