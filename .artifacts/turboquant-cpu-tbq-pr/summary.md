# TurboQuant CPU TBQ PR Results

Model: `Qwen3.5-4B-Q4_K_M.gguf`

Settings: CPU only, 4 threads, `flash_attn=on`, `llama-bench` `pp32/tg8`, `llama-perplexity` on `wikitext-2-raw/wiki.test.raw` with `ctx=256`, `chunks=4`.

## Benchmark Table

| Cache type | Prompt t/s | Gen t/s | KV MiB | Compression vs f16 | PPL | KLD | RMS Δp | Same top p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` | 50.67 | 15.72 | 64.00 | 1.00x | 13.8387 | 0.00000 | 0.000% | 100.000% |
| `q8_0` | 50.63 | 15.67 | 34.00 | 1.88x | 13.8348 | 0.00320 | 1.510% | 97.835% |
| `q4_0` | 50.46 | 15.64 | 18.00 | 3.56x | 13.8400 | 0.00912 | 2.179% | 93.898% |
| `tbq3_0` | 46.19 | 8.29 | 12.25 | 5.22x | 14.3198 | 0.02647 | 4.471% | 91.732% |
| `tbq4_0` | 45.84 | 8.31 | 16.25 | 3.94x | 13.8323 | 0.00960 | 2.892% | 94.094% |

## Key observations

- `tbq4_0` is the best-balanced TurboQuant variant in this CPU-only sweep: `3.94x` KV compression with KLD close to `q4_0`.
- `tbq3_0` delivers the smallest KV cache at `5.22x` compression vs `f16`, with a larger quality tradeoff.
- `tbq4_0` reduces KV cache below stock `q4_0` while keeping similar KLD and better perplexity in this run.
- These numbers are taken from the TBQ-only branch after removing the wider TBQP surface from the first PR scope.
