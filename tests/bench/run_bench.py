#!/usr/bin/env python3
"""Run the native runtime's benchmark mode and write a timestamped Markdown report.

Times every distinct NPU kernel shape (ms + GMAC/s) and an end-to-end
prefill/decode, then appends the run to a trend table so improvements and
regressions are visible over time.

Usage:
    python3 tests/bench/run_bench.py [weights_dir]

Requires only the standard library. The runtime binary must be built
(runtime/cpp/build/alveare_runtime) and the NPU stack available at runtime
(system XRT — no conda/env_setup needed just to run).
"""
import datetime
import os
import pathlib
import re
import socket
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = ROOT / "runtime/cpp/build/alveare_runtime"
MANIFEST = ROOT / "kernels/build/manifest.json"
REPORTS_DIR = ROOT / "benchmarks" / "reports"
TREND_FILE = ROOT / "benchmarks" / "README.md"


def git(*args, default="?"):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return default


def run_binary(weights_dir):
    env = dict(os.environ, ALVEARE_BENCH="1")
    print(f"Running benchmark (loads {weights_dir}, ~5 min)...", flush=True)
    proc = subprocess.run(
        [str(BIN), str(weights_dir), str(MANIFEST), "8199"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if "=== ALVEARE BENCH END ===" not in proc.stdout:
        sys.stderr.write(proc.stdout[-4000:])
        raise SystemExit(f"benchmark did not complete (rc={proc.returncode})")
    return proc.stdout


def parse(out):
    kernels = []
    for m in re.finditer(
        r"^KERNEL (\S+) (\S+) (\d+) (\d+) ([\d.]+) ([\d.]+)", out, re.M
    ):
        kind, label, N, K, ms, gmacs = m.groups()
        kernels.append(dict(kind=kind, label=label, N=int(N), K=int(K),
                            ms=float(ms), gmacs=float(gmacs)))

    meta = {}
    mm = re.search(r"BENCH_META model_type=(\S+) layers=(\d+)", out)
    if mm:
        meta["model_type"], meta["layers"] = mm.group(1), int(mm.group(2))

    e2e = {}
    # New log ("prefilling N new of M prompt tokens") or the older
    # ("Starting prefill of M tokens"); use the total prompt-token count M.
    m = (re.search(r"prefilling \d+ new of (\d+) prompt tokens", out)
         or re.search(r"Starting prefill of (\d+) tokens", out))
    if m:
        e2e["prefill_tokens"] = int(m.group(1))
    m = re.search(r"Prefill completed in ([\d.]+)s", out)
    if m:
        e2e["prefill_s"] = float(m.group(1))

    toks = [tuple(map(float, m.groups())) for m in re.finditer(
        r"Token \d+/\d+ in ([\d.]+)ms \[ffn=([\d.]+) gemv=([\d.]+) "
        r"lm_head=([\d.]+) cpu=([\d.]+)", out)]
    dec = toks[1:] if len(toks) > 1 else toks  # drop first (warm-up) sample
    if dec:
        n = len(dec)
        keys = ["step", "ffn", "gemv", "lm_head", "cpu"]
        for i, k in enumerate(keys):
            e2e[f"decode_{k}_ms"] = sum(t[i] for t in dec) / n
        e2e["decode_samples"] = n
    return meta, kernels, e2e


def write_report(meta, kernels, e2e):
    now = datetime.datetime.now()
    sha = git("rev-parse", "--short", "HEAD")
    subj = git("log", "-1", "--pretty=%s")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    host = socket.gethostname()
    stamp = now.strftime("%Y%m%d-%H%M%S")

    lines = [
        f"# Benchmark — {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- **Commit**: `{sha}` ({branch}) — {subj}",
        f"- **Host**: {host}",
        f"- **Model**: {meta.get('model_type', '?')}, "
        f"{meta.get('layers', '?')} layers",
        "",
        "## Kernels (avg over 20, resident weights)",
        "",
        "| kernel | shape (N×K) | ms | GMAC/s |",
        "|---|---|---:|---:|",
    ]
    for k in kernels:
        lines.append(f"| {k['kind']} {k['label']} | {k['N']}×{k['K']} | "
                     f"{k['ms']:.2f} | {k['gmacs']:.1f} |")

    lines += ["", "## End-to-end (gemma4, greedy)", ""]
    if "prefill_s" in e2e:
        pt = e2e.get("prefill_tokens")
        per = f" ({e2e['prefill_s'] / pt:.2f}s/token)" if pt else ""
        lines.append(f"- **Prefill**: {e2e['prefill_s']:.2f}s"
                     f" for {pt} tokens{per}")
    if "decode_step_ms" in e2e:
        lines.append(
            f"- **Decode**: {e2e['decode_step_ms']:.1f} ms/token "
            f"(avg of {e2e['decode_samples']}) — "
            f"ffn={e2e['decode_ffn_ms']:.1f} gemv={e2e['decode_gemv_ms']:.1f} "
            f"lm_head={e2e['decode_lm_head_ms']:.1f} cpu={e2e['decode_cpu_ms']:.1f}")
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{stamp}-{sha}.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path.relative_to(ROOT)}")

    # Trend row: date | sha | decode ms/tok | prefill s/tok | fused GMAC/s | lm_head GMAC/s
    fused = next((k["gmacs"] for k in kernels if k["kind"] == "ffn_fused"), None)
    lmh = next((k["gmacs"] for k in kernels if k["label"] == "lm_head"), None)
    dec = e2e.get("decode_step_ms")
    pt = e2e.get("prefill_tokens")
    pptok = (e2e["prefill_s"] / pt) if (pt and "prefill_s" in e2e) else None
    fmt = lambda v, p: (f"{v:.{p}f}" if v is not None else "?")
    row = (f"| {now.strftime('%Y-%m-%d %H:%M')} | `{sha}` | "
           f"{fmt(dec, 0)} | {fmt(pptok, 2)} | {fmt(fused, 1)} | {fmt(lmh, 1)} | "
           f"[report]({report_path.relative_to(ROOT.joinpath('benchmarks'))}) |")

    header = [
        "# Benchmark history",
        "",
        "Newest first. Each row is one `run_bench.py` execution; see the linked "
        "report for the full per-kernel breakdown.",
        "",
        "| date | commit | decode ms/tok | prefill s/tok | FFN GMAC/s | "
        "lm_head GMAC/s | report |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    existing = []
    if TREND_FILE.exists():
        body = TREND_FILE.read_text().splitlines()
        existing = [l for l in body if l.startswith("| ") and "---" not in l
                    and not l.startswith("| date")]
    TREND_FILE.write_text("\n".join(header + [row] + existing) + "\n")
    print(f"Updated {TREND_FILE.relative_to(ROOT)}")


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "quantized_weights_gemma4")
    if not BIN.exists():
        raise SystemExit(f"binary not built: {BIN}")
    out = run_binary(weights)
    meta, kernels, e2e = parse(out)
    if not kernels:
        sys.stderr.write(out[-4000:])
        raise SystemExit("no kernel timings parsed")
    write_report(meta, kernels, e2e)


if __name__ == "__main__":
    main()
