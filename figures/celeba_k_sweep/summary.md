# CelebA K-sweep (2000 iterations per run)

## PSNR (dB) — higher is better

| method | K=64 | K=256 | K=512 | K=1024 |
| --- | --- | --- | --- | --- |
| vanilla | 21.74 | 22.81 | 22.81 | 23.37 |
| simvq | 18.60 | 19.58 | 19.45 | 20.05 |
| drift | 20.36 | 21.18 | 21.51 | 21.85 |

## Perplexity — max possible value is K (uniform code use)

| method | K=64 | K=256 | K=512 | K=1024 |
| --- | --- | --- | --- | --- |
| vanilla | 53.6 | 181.3 | 387.0 | 652.1 |
| simvq | 6.8 | 15.0 | 20.4 | 43.5 |
| drift | 61.5 | 231.3 | 427.1 | 787.5 |

## Utilization (%) — fraction of codes with nonzero use

| method | K=64 | K=256 | K=512 | K=1024 |
| --- | --- | --- | --- | --- |
| vanilla | 100.0% | 100.0% | 99.8% | 99.6% |
| simvq | 95.3% | 93.0% | 92.6% | 90.6% |
| drift | 100.0% | 100.0% | 100.0% | 99.8% |
