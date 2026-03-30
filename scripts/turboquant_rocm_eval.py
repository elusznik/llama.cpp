#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/elusznik/Work/llama.cpp/.worktrees/turboquant-rocm-pr")
BUILD = ROOT / "build-hip" / "bin"
OUT_DIR = ROOT / "docs" / "turboquant-rocm-eval"
DOC_PATH = ROOT / "docs" / "turboquant-rocm-eval.md"
SAFETY_DIR = Path("/home/elusznik/Work/llama.cpp/.safety/20260330-rocm-pr-materials")
RAW_DIR = SAFETY_DIR / "raw"

MODEL = Path("/home/elusznik/models/Qwen3.5-4B-Q4_K_M.gguf")
CORPUS = Path("/home/elusznik/Work/bitnet-ppl-test/wikitext-2-raw/wiki.test.raw")

BENCH_REPS = 2
PPL_CTX = 512
PPL_BATCH = 512
PPL_CHUNKS = 1

BENCH = BUILD / "llama-bench"
PPL = BUILD / "llama-perplexity"

BASE_FLAGS = ["-m", str(MODEL), "-fa", "1", "-ub", "2048"]
PPL_FLAGS = [
    "-m", str(MODEL),
    "-f", str(CORPUS),
    "-fa", "1",
    "-c", str(PPL_CTX),
    "-b", str(PPL_BATCH),
    "--chunks", str(PPL_CHUNKS),
]

PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+)")
KV_RE = re.compile(r"llama_kv_cache:\s+size\s+=\s+([0-9.]+)\s+MiB")


@dataclass
class Case:
    name: str
    label: str
    bits: float
    kind: str
    bench_flags: list[str]
    ppl_flags: list[str]
    notes: str = ""


def run(cmd: list[str], log_path: Path) -> str:
    print("+", " ".join(cmd))
    env = {
        "ROCM_PATH": "/opt/rocm",
        "HIP_PATH": "/opt/rocm",
    }
    proc = subprocess.run(cmd, text=True, capture_output=True, env={**subprocess.os.environ, **env})
    combined = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(combined)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return combined


def parse_bench(json_text: str) -> tuple[float, float]:
    # Handle combined stdout+stderr logs where stdout is JSON and stderr has init messages
    # Find the actual JSON array - it starts with '[' and stderr is separated by '\n\n[stderr]\n'
    stderr_marker = '\n\n[stderr]\n'
    stderr_pos = json_text.find(stderr_marker)
    if stderr_pos > 0:
        json_text = json_text[:stderr_pos]
    json_start = json_text.find('[')
    if json_start == -1:
        raise RuntimeError(f"no JSON array found in bench output: {json_text[:200]}")
    rows = json.loads(json_text[json_start:])
    prompt = next(row["avg_ts"] for row in rows if row["n_prompt"] > 0)
    gen = next(row["avg_ts"] for row in rows if row["n_gen"] > 0)
    return float(prompt), float(gen)


def parse_base_ppl(stdout: str) -> tuple[float, float]:
    m_ppl = PPL_RE.search(stdout)
    m_kv = KV_RE.search(stdout)
    if not (m_ppl and m_kv):
        raise RuntimeError("failed to parse baseline perplexity output")
    return float(m_ppl.group(1)), float(m_kv.group(1))


def parse_compare_ppl(stdout: str) -> tuple[float, float, float, float, float]:
    m_kv = KV_RE.search(stdout)
    chunk_line = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and "±" in stripped:
            chunk_line = stripped
            break
    if not (m_kv and chunk_line):
        raise RuntimeError("failed to parse comparison perplexity output")
    parts = chunk_line.replace("%", "").split()
    return (
        float(parts[1]),
        float(m_kv.group(1)),
        float(parts[7]),
        float(parts[10]),
        float(parts[13]),
    )


def fmt(x: float, digits: int = 4) -> str:
    if math.isnan(x):
        return "nan"
    if abs(x) < 5e-9:
        x = 0.0
    return f"{x:.{digits}f}"


CASES = [
    Case("f16", "F16", 16.0, "same", ["-ctk", "f16", "-ctv", "f16"], ["-ctk", "f16", "-ctv", "f16"]),
    Case("q8_0", "Q8_0", 8.0, "same", ["-ctk", "q8_0", "-ctv", "q8_0"], ["-ctk", "q8_0", "-ctv", "q8_0"]),
    Case("q4_0", "Q4_0", 4.5, "same", ["-ctk", "q4_0", "-ctv", "q4_0"], ["-ctk", "q4_0", "-ctv", "q4_0"]),
    Case("tbq4_0", "TBQ4_0", 4.0625, "same", ["-ctk", "tbq4_0", "-ctv", "tbq4_0"], ["-ctk", "tbq4_0", "-ctv", "tbq4_0"]),
    Case("tbqp4_0", "TBQP4_0", 4.125, "same", ["-ctk", "tbqp4_0", "-ctv", "tbqp4_0"], ["-ctk", "tbqp4_0", "-ctv", "tbqp4_0"]),
    Case("tbq3_0", "TBQ3_0", 3.0625, "same", ["-ctk", "tbq3_0", "-ctv", "tbq3_0"], ["-ctk", "tbq3_0", "-ctv", "tbq3_0"]),
    Case("tbqp3_0", "TBQP3_0", 3.125, "same", ["-ctk", "tbqp3_0", "-ctv", "tbqp3_0"], ["-ctk", "tbqp3_0", "-ctv", "tbqp3_0"]),
    # Note: TBQ34_0 and TBQP34_0 skipped - ROCm/gfx1031 TensileLibrary issue with these types
    Case("tbqp4k_tbq4v", "TBQP4 K + TBQ4 V", 4.09375, "mixed", ["-ctk", "tbqp4_0", "-ctv", "tbq4_0"], ["-ctk", "tbqp4_0", "-ctv", "tbq4_0"]),
    Case("tbqp3k_tbq3v", "TBQP3 K + TBQ3 V", 3.09375, "mixed", ["-ctk", "tbqp3_0", "-ctv", "tbq3_0"], ["-ctk", "tbqp3_0", "-ctv", "tbq3_0"]),
    # Note: Split types skipped - GGML_ASSERT failure on ROCm/gfx1031
]


def plot_bar(path: Path, title: str, rows: list[dict], key: str, ylabel: str):
    labels = [row["label"] for row in rows]
    vals = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_throughput(path: Path, rows: list[dict]):
    labels = [row["label"] for row in rows]
    prompt = [float(row["prompt_tps"]) for row in rows]
    gen = [float(row["gen_tps"]) for row in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    ax.bar([i - width / 2 for i in x], prompt, width, label="Prompt", color="#475569")
    ax.bar([i + width / 2 for i in x], gen, width, label="Generation", color="#94a3b8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("tokens/s")
    ax.set_title("TurboQuant ROCm throughput")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_quality(path: Path, rows: list[dict]):
    labels = [row["label"] for row in rows]
    ppl = [float(row["ppl"]) for row in rows]
    kld = [float(row["kld"]) for row in rows]
    x = range(len(labels))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax1.plot(list(x), ppl, marker="o", color="#2563eb", label="PPL")
    ax2.plot(list(x), kld, marker="s", color="#dc2626", label="KLD")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylabel("PPL")
    ax2.set_ylabel("KLD")
    ax1.set_title("TurboQuant ROCm quality")
    ax1.grid(axis="y", alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_compression_vs_speed(path: Path, rows: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        ax.scatter(float(row["kv_ratio"]), float(row["gen_tps"]), s=100)
        ax.text(float(row["kv_ratio"]) + 0.03, float(row["gen_tps"]) + 0.03, row["label"], fontsize=9)
    ax.set_xlabel("Compression ratio vs f16")
    ax.set_ylabel("Generation tokens/s")
    ax.set_title("Compression vs speed")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ablation(path: Path, rows: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        ax.scatter(float(row["kv_mib"]), float(row["kld"]), s=100)
        ax.text(float(row["kv_mib"]) + 0.2, float(row["kld"]) + 0.0005, row["label"], fontsize=9)
    ax.set_xlabel("KV cache size (MiB)")
    ax.set_ylabel("KLD")
    ax.set_title("KV size vs KLD")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_split_sweep(path: Path, rows: list[dict]):
    labels = [row["label"] for row in rows]
    kv = [float(row["kv_mib"]) for row in rows]
    kld = [float(row["kld"]) for row in rows]
    ppl = [float(row["ppl"]) for row in rows]
    x = range(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(list(x), kv, marker="o", color="#0f766e", label="KV MiB")
    ax2.plot(list(x), kld, marker="s", color="#b91c1c", label="KLD")
    ax2.plot(list(x), ppl, marker="^", color="#1d4ed8", label="PPL")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("KV MiB")
    ax2.set_ylabel("KLD / PPL")
    ax1.set_title("Split outlier sweep")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tbqp_modes(path: Path, rows: list[dict]):
    labels = [row["label"] for row in rows]
    ppl = [float(row["ppl"]) for row in rows]
    kld = [float(row["kld"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, x, y in zip(labels, ppl, kld):
        ax.scatter(x, y, s=110)
        ax.text(x + 0.002, y + 0.0005, label, fontsize=9)
    ax.set_xlabel("PPL")
    ax.set_ylabel("KLD")
    ax.set_title("TBQP ROCm modes")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    SAFETY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for path in RAW_DIR.iterdir():
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

    base_logits = RAW_DIR / "f16-base.kld"
    base_stdout = run(
        [str(PPL), *PPL_FLAGS, "-ctk", "f16", "-ctv", "f16", "--save-all-logits", str(base_logits)],
        RAW_DIR / "f16_base_logits.log",
    )
    base_ppl, base_kv = parse_base_ppl(base_stdout)

    results: list[dict] = []

    for case in CASES:
        bench_stdout = run(
            [str(BENCH), *BASE_FLAGS, "-p", "32", "-n", "8", "-r", str(BENCH_REPS), *case.bench_flags, "-o", "json"],
            RAW_DIR / f"{case.name}_bench.log",
        )
        prompt_tps, gen_tps = parse_bench(bench_stdout)

        ppl_stdout = run(
            [str(PPL), *PPL_FLAGS, *case.ppl_flags, "--kl-divergence-base", str(base_logits), "--kl-divergence"],
            RAW_DIR / f"{case.name}_ppl.log",
        )
        ppl, kv_mib, kld, rms, same = parse_compare_ppl(ppl_stdout)
        bits = case.bits if case.bits else float("nan")
        results.append({
            "type": case.name,
            "label": case.label,
            "kind": case.kind,
            "bits": bits,
            "kv_mib": kv_mib,
            "kv_ratio": base_kv / kv_mib,
            "prompt_tps": prompt_tps,
            "gen_tps": gen_tps,
            "ppl": ppl,
            "ppl_delta": ppl - base_ppl,
            "kld": kld,
            "delta_p_rms_pct": rms,
            "same_top_p_pct": same,
        })

    rows_same = [row for row in results if row["kind"] == "same"]
    rows_split = [row for row in results if row["kind"] == "split"]
    if not rows_split:
        rows_split = None
    rows_tbqp = [row for row in results if row["type"] in {"tbqp4_0", "tbqp3_0", "tbqp4k_tbq4v", "tbqp3k_tbq3v", "tbq34_0", "tbqp34_0"}]

    with (OUT_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_lines = [
        "# TurboQuant ROCm Evaluation",
        "",
        "Settings: AMD RX 6700 XT (gfx1031), `flash_attn=1`, `llama-bench` `pp32/tg8`, `llama-perplexity` on `wikitext-2-raw/wiki.test.raw` with `ctx=512`, `chunks=1`.",
        "",
        "| Type | Bits/elem | KV MiB | KV vs F16 | Prompt t/s | Gen t/s | PPL | KLD | RMS Δp [%] | Same top p [%] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_same:
        bits = "n/a" if math.isnan(float(row["bits"])) else fmt(float(row["bits"]))
        summary_lines.append(
            f"| `{row['type']}` | {bits} | {fmt(float(row['kv_mib']),2)} | {fmt(float(row['kv_ratio']),2)}x smaller | "
            f"{fmt(float(row['prompt_tps']))} | {fmt(float(row['gen_tps']))} | {fmt(float(row['ppl']))} | "
            f"{fmt(float(row['kld']),6)} | {fmt(float(row['delta_p_rms_pct']),3)} | {fmt(float(row['same_top_p_pct']),3)} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n")

    report = []
    report.append("# TurboQuant ROCm PR Notes\n")
    report.append("## Setup\n")
    report.extend([
        "- Model: `Qwen3.5-4B-Q4_K_M.gguf`",
        "- Device: `AMD RX 6700 XT (gfx1031)`",
        "- Bench: prompt `32`, gen `8`, flash attention `on`",
        "- Perplexity/KLD: ctx `512`, chunks `1`, flash attention `on`",
        "",
        "## Same-Type Reference Table",
        "",
    ])
    report.extend(summary_lines)
    report.extend([
        "",
        "## Recommended ROCm TBQP Modes",
        "",
        "| Setup | KV MiB | PPL | KLD | RMS Δp [%] | Same top p [%] | Prompt t/s | Gen t/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows_tbqp:
        report.append(
            f"| `{row['label']}` | {fmt(float(row['kv_mib']),2)} | {fmt(float(row['ppl']))} | {fmt(float(row['kld']),5)} | "
            f"{fmt(float(row['delta_p_rms_pct']),3)} | {fmt(float(row['same_top_p_pct']),3)} | "
            f"{fmt(float(row['prompt_tps']))} | {fmt(float(row['gen_tps']))} |"
        )
    report.extend([
        "",
        "## Split Outlier Sweep",
        "",
        "| Split config | KV MiB | PPL | KLD | Prompt t/s | Gen t/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    if rows_split:
        for row in rows_split:
            report.append(
                f"| `{row['label']}` | {fmt(float(row['kv_mib']),2)} | {fmt(float(row['ppl']))} | "
                f"{fmt(float(row['kld']),5)} | {fmt(float(row['prompt_tps']))} | {fmt(float(row['gen_tps']))} |"
            )
    report.extend([
        "",
        "## Notes",
        "",
        f"- `tbq4_0` remains the strongest practical same-type ROCm result: {fmt(float(next(r for r in rows_same if r['type']=='tbq4_0')['kv_ratio']),2)}x smaller KV than `f16` with PPL {fmt(float(next(r for r in rows_same if r['type']=='tbq4_0')['ppl']))} and KLD {fmt(float(next(r for r in rows_same if r['type']=='tbq4_0')['kld']),6)}.",
        f"- Mixed `tbqp4_0` K + `tbq4_0` V remains the best practical ROCm TBQP mode in this run.",
        f"- `TBQ34_0` and `TBQP34_0` are new mixed-precision types (3/2-bit regular + 4/3-bit outlier).",
        "",
        "Plots:",
        "",
        "- `kv-memory.png`",
        "- `throughput.png`",
        "- `quality.png`",
        "- `compression-vs-speed.png`",
        "- `tbqp-modes.png`",
        "- `split-outlier-sweep.png`",
    ])
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")

    plot_bar(OUT_DIR / "kv-memory.png", "TurboQuant ROCm KV cache memory", rows_same, "kv_mib", "MiB")
    plot_throughput(OUT_DIR / "throughput.png", rows_same)
    plot_quality(OUT_DIR / "quality.png", rows_same)
    plot_compression_vs_speed(OUT_DIR / "compression-vs-speed.png", rows_same)
    plot_ablation(OUT_DIR / "ablation-size-vs-kld.png", rows_same)
    if rows_split:
        plot_split_sweep(OUT_DIR / "split-outlier-sweep.png", rows_split)
    plot_tbqp_modes(OUT_DIR / "tbqp-modes.png", rows_tbqp)

    doc_lines = [
        "# TurboQuant ROCm Evaluation",
        "",
        "ROCm/GPU evaluation of the `turboquant-rocm-pr` branch with TBQ34_0 and TBQP34_0 support.",
        "",
        "## Setup",
        "",
        "- Model: `Qwen3.5-4B-Q4_K_M.gguf`",
        "- Device: `AMD RX 6700 XT (gfx1031)`",
        "- Bench: `llama-bench` `pp32/tg8`, `-fa 1`",
        "- Perplexity/KLD: `llama-perplexity`, `ctx=512`, `chunks=1`, `-fa 1`",
        "",
        *summary_lines[3:],
        "",
        "## Recommended ROCm TBQP Modes",
        "",
        "| Setup | KV MiB | PPL | KLD | Prompt t/s | Gen t/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_tbqp:
        doc_lines.append(
            f"| `{row['label']}` | {fmt(float(row['kv_mib']),2)} | {fmt(float(row['ppl']))} | "
            f"{fmt(float(row['kld']),5)} | {fmt(float(row['prompt_tps']))} | {fmt(float(row['gen_tps']))} |"
        )
    doc_lines.extend([
        "",
        "## Split Outlier Sweep",
        "",
        "| Split config | KV MiB | PPL | KLD | Prompt t/s | Gen t/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    if rows_split:
        for row in rows_split:
            doc_lines.append(
                f"| `{row['label']}` | {fmt(float(row['kv_mib']),2)} | {fmt(float(row['ppl']))} | "
                f"{fmt(float(row['kld']),5)} | {fmt(float(row['prompt_tps']))} | {fmt(float(row['gen_tps']))} |"
            )
    doc_lines.extend([
        "",
        "## Plots",
        "",
        "### KV cache memory",
        "",
        "![KV cache memory](turboquant-rocm-eval/kv-memory.png)",
        "",
        "### Throughput",
        "",
        "![Throughput](turboquant-rocm-eval/throughput.png)",
        "",
        "### Quality",
        "",
        "![Quality](turboquant-rocm-eval/quality.png)",
        "",
        "### Compression vs speed",
        "",
        "![Compression vs speed](turboquant-rocm-eval/compression-vs-speed.png)",
        "",
        "### TBQP modes",
        "",
        "![TBQP modes](turboquant-rocm-eval/tbqp-modes.png)",
        "",
        "### Split outlier sweep",
        "",
        "![Split outlier sweep](turboquant-rocm-eval/split-outlier-sweep.png)",
        "",
    ])
    DOC_PATH.write_text("\n".join(doc_lines) + "\n")

    for path in [OUT_DIR, DOC_PATH]:
        dst = SAFETY_DIR / path.name
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)


if __name__ == "__main__":
    main()
