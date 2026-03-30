I spent part of the weekend experimenting with this — ended up going pretty deep. TurboQuant's KV cache compression is genuinely interesting but it had no NEON SIMD support on ARM, so on Apple Silicon it was falling back to generic C loops and leaving a lot of performance on the table.

Wrote optimized NEON kernels for both `tbq4_0` and `tbq3_0` — the vec_dot path is tricky because TBQ applies a random rotation matrix before quantizing, so you can't use the usual int8 dot product shortcuts. You have to dequantize through an inverse rotation in float32. Got tbq4_0 from ~50 t/s up to ~210+ t/s on 32K context on an M4 Mac mini (16GB unified memory, 4 threads), testing against Qwen3.5-4B Q4_K_M. Closed the gap to q4_0 from ~50 t/s down to ~16 t/s.

Also found and fixed an `arch-fallback.h` linker bug that was blocking ARM64 builds entirely.

Posted the kernels + benchmark data as a comment on the upstream PR:
https://github.com/ggml-org/llama.cpp/pull/21089#issuecomment-4152369277

Fork with the implementation:
https://github.com/CuriosityQuantified/llama.cpp/tree/neon-arm-optimization
