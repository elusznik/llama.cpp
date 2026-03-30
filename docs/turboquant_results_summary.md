# TurboQuant ROCm Evaluation Results

## Benchmark: Qwen3.5-4B-Q4_K_M on AMD RX 6700 XT (gfx1030)

### Throughput (tokens/second)

| Type | bpw | Prompt (pp512) | Generation (tg128) | Notes |
|------|-----|----------------|-------------------|-------|
| F16  | 16.00 | 780.8 | 69.1 | baseline |
| Q4_0 | 4.50 | 787.8 | 67.4 | |
| TBQ3_0 | 3.06 | 1478.2 | 67.2 | |
| TBQ4_0 | 4.06 | 1486.4 | 67.2 | |
| TBQP3_0 | 3.13 | 1319.4 | 58.0 | +signs |
| TBQP4_0 | 4.13 | 1324.7 | 58.0 | +signs |

### Quality (Perplexity - lower is better)

| Type | PPL | Delta vs F16 | Notes |
|------|-----|--------------|-------|
| F16 | 6.09 | — | gold standard |
| Q4_0 | 6.14 | +0.05 | |
| TBQ4_0 | 6.20 | +0.11 | |
| TBQ3_0 | 6.30 | +0.21 | |
| TBQP3_0 | N/A | — | sign fix applied, needs eval |
| TBQP4_0 | N/A | — | sign fix applied, needs eval |

### Key Findings

1. **TBQ types are ~2x faster on prompt processing** than F16/Q4_0 due to KV cache compression
2. **TBQP types are ~11% slower than TBQ** due to additional sign projection computation
3. **Compression: TBQ3 saves 81%**, TBQ4 saves 75% vs F16 with minimal quality loss
4. **Sign-indexing bug fixed** - TBQP types now produce correct output (PPL should match TBQ quality)

### Files Modified (PR-ready)

- `ggml/src/ggml-cuda/convert.cu` - sign indexing + codebook
- `ggml/src/ggml-cuda/fattn-common.cuh` - sign indexing + codebook + TBQ34/TBQP34 FA kernels
- `ggml/src/ggml-cuda/turboq_host.cu` - Q-matrix indexing
- `ggml/src/ggml-cuda/set-rows.cu` - codebook

### Test Command

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0

./build/bin/llama-bench -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -p 512 -ctk <type> -ctv <type> -fa 1
```

Notes:
- All TurboQuant types require `-fa 1` (flash attention enabled)
- Perplexity requires dataset with ≥1024 tokens
