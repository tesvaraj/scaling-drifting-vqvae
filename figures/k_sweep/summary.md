# CIFAR-10 K-sweep (2000 iterations per run)

## PSNR (dB) — higher is better

| method | K=64 | K=256 | K=512 | K=1024 |
| --- | --- | --- | --- | --- |
| vanilla | 22.26 | 23.42 | 23.81 | 24.37 |
| simvq | 19.10 | 20.03 | 20.49 | 20.94 |
| drift | 20.87 | 21.71 | 22.12 | 22.43 |

## Perplexity — max possible value is K (uniform code use)

| method | K=64 | K=256 | K=512 | K=1024 |
| --- | --- | --- | --- | --- |
| vanilla | 46.6 | 182.2 | 374.2 | 678.6 |
| simvq | 8.5 | 22.4 | 29.3 | 54.0 |
| drift | 61.2 | 229.9 | 421.6 | 758.6 |

## Utilization (%) — fraction of codes with nonzero use

| method | K=64 | K=256 | K=512 | K=1024 |
| --- | --- | --- | --- | --- |
| vanilla | 100.0% | 100.0% | 100.0% | 100.0% |
| simvq | 98.4% | 97.3% | 89.8% | 90.4% |
| drift | 100.0% | 100.0% | 100.0% | 100.0% |
