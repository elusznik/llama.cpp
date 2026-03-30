# TurboQuant ROCm Evaluation

ROCm/GPU evaluation of the `turboquant-rocm-pr` branch with TBQ34_0 and TBQP34_0 support.

## Setup

- Model: `Qwen3.5-4B-Q4_K_M.gguf`
- Device: `AMD RX 6700 XT (gfx1031)`
- Bench: `llama-bench` `pp32/tg8`, `-fa 1`
- Perplexity/KLD: `llama-perplexity`, `ctx=512`, `chunks=1`, `-fa 1`


| Type | Bits/elem | KV MiB | KV vs F16 | Prompt t/s | Gen t/s | PPL | KLD | RMS Δp [%] | Same top p [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` | 16.0000 | 16.00 | 1.00x smaller | 780.7875 | 69.0688 | 6.0871 | 0.000000 | 0.001 | 100.000 |
| `q8_0` | 8.0000 | 8.50 | 1.88x smaller | 786.4767 | 68.0767 | 6.0579 | 0.001980 | 1.340 | 98.039 |
| `q4_0` | 4.5000 | 4.50 | 3.56x smaller | 787.8366 | 67.4458 | 6.1376 | 0.006170 | 2.181 | 97.255 |
| `tbq4_0` | 4.0625 | 4.06 | 3.94x smaller | 743.4179 | 64.8070 | 6.2033 | 0.006810 | 2.162 | 96.078 |
| `tbqp4_0` | 4.1250 | 4.12 | 3.88x smaller | 707.6313 | 56.4829 | 51.1985 | 1.934600 | 49.490 | 45.490 |
| `tbq3_0` | 3.0625 | 3.06 | 5.23x smaller | 772.3775 | 65.9301 | 6.3037 | 0.019420 | 4.286 | 93.333 |
| `tbqp3_0` | 3.1250 | 3.12 | 5.13x smaller | 708.4094 | 56.5872 | 56.6775 | 2.083690 | 50.689 | 46.667 |

## Recommended ROCm TBQP Modes

| Setup | KV MiB | PPL | KLD | Prompt t/s | Gen t/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TBQP4_0` | 4.12 | 51.1985 | 1.93460 | 707.6313 | 56.4829 |
| `TBQP3_0` | 3.12 | 56.6775 | 2.08369 | 708.4094 | 56.5872 |
| `TBQP4 K + TBQ4 V` | 4.09 | 20.9492 | 1.05405 | 740.0787 | 60.7368 |
| `TBQP3 K + TBQ3 V` | 3.09 | 22.7890 | 1.10416 | 724.0093 | 59.2994 |

## Split Outlier Sweep

| Split config | KV MiB | PPL | KLD | Prompt t/s | Gen t/s |
| --- | ---: | ---: | ---: | ---: | ---: |

## Plots

### KV cache memory

![KV cache memory](turboquant-rocm-eval/kv-memory.png)

### Throughput

![Throughput](turboquant-rocm-eval/throughput.png)

### Quality

![Quality](turboquant-rocm-eval/quality.png)

### Compression vs speed

![Compression vs speed](turboquant-rocm-eval/compression-vs-speed.png)

### TBQP modes

![TBQP modes](turboquant-rocm-eval/tbqp-modes.png)

### Split outlier sweep

![Split outlier sweep](turboquant-rocm-eval/split-outlier-sweep.png)

