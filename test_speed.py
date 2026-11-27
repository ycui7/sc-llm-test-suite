#!/usr/bin/env python3
"""Check 3: speed benchmark (pp2048 / tg512, concurrency 1, 2, 4).

Prefers llama-benchy when available (canonical numbers, tokenizer-exact):
  llama-benchy --base-url ... --model ... --pp 2048 --tg 512
  --runs 3 --concurrency 1 2 4 --format json --save-result <tmp>.json
  (set QC_BENCHY=/path/to/llama-benchy to override discovery)
If llama-benchy is not on PATH, falls back to a stdlib-only round-robin
benchmark: N parallel chat requests with a long shared prompt (~2048 tokens
of filler) requesting a fixed-length generation; reports per-concurrency
aggregate decode tok/s from usage.completion_tokens. Fallback numbers are
directional, not tokenizer-exact.

Zero-interference rule: run this alone against the endpoint, don't poll.

Concurrency gate (2026-11-26): aggregate tg at c2 must exceed
CONC_MIN_RATIO x aggregate tg at c1, else the endpoint is claimed to have
no real concurrency: the bench is aborted early and the test FAILS.

Usage:
  python3 test_speed.py --model ds4f-gb10 [--pp 2048] [--tg 512]
                        [--runs 3] [--concurrency 1 2 4]
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time

from qc_common import chat, http_json, make_arg_parser, resolve_model, Result

BENCHY_CANDIDATES = [
    os.environ.get("QC_BENCHY", ""),          # explicit override
    os.path.expanduser("~/.venv/bin/llama-benchy"),
    shutil.which("llama-benchy") or "",
]


CONC_MIN_RATIO = 1.5  # c2 aggregate tg must EXCEED this x c1 aggregate tg


def run_benchy(args, tmp_json, concurrency):
    benchy = next((c for c in BENCHY_CANDIDATES if c and os.path.exists(c)), None)
    if not benchy:
        return None
    cmd = [benchy,
           "--base-url", args.base_url,
           "--model", args.model,
           "--served-model-name", args.model,
           "--api-key", args.api_key or "EMPTY",
           "--pp", str(args.pp), "--tg", str(args.tg),
           "--no-cache", "--depth", "0",
           "--latency-mode", "generation",
           "--no-adapt-prompt",
           "--runs", str(args.runs),
           "--concurrency", *[str(c) for c in concurrency],
           "--format", "json",
           "--save-result", tmp_json]
    print("running: " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=args.benchy_timeout)
    if proc.returncode != 0 or not os.path.exists(tmp_json):
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
        return None
    with open(tmp_json) as f:
        return json.load(f)


def benchy_matrix(data):
    rows = []
    for b in data.get("benchmarks", []):
        rows.append({
            "concurrency": b.get("concurrency"),
            "pp_tps": (b.get("pp_throughput") or {}).get("mean"),
            "tg_tps_agg": (b.get("tg_throughput") or {}).get("mean"),
            "tg_tps_req": (b.get("tg_req_throughput") or {}).get("mean"),
            "ttfr_s": (b.get("ttfr") or {}).get("mean", 0) / 1000.0,
        })
    return rows


def conc_gate(rows, requested):
    """Concurrency scaling gate. Returns (ok, detail) or None if not applicable.

    Applies only when both c1 and c2 were requested AND measured. Aggregate
    decode rate at c2 must EXCEED CONC_MIN_RATIO x c1, else the endpoint
    serializes requests (no real concurrency).
    """
    if 1 not in requested or 2 not in requested:
        return None
    by = {row["concurrency"]: row for row in rows}
    if 1 not in by or 2 not in by:
        return None
    a1 = by[1].get("tg_tps_agg") or 0
    a2 = by[2].get("tg_tps_agg") or 0
    if a1 <= 0:
        return (False, f"c1 agg={a1:.1f} t/s (cannot scale from zero)")
    ratio = a2 / a1
    return (ratio > CONC_MIN_RATIO,
            f"c2/c1 agg tg {a2:.1f}/{a1:.1f} = {ratio:.2f}x (need >{CONC_MIN_RATIO}x)")


def fallback_bench(args):
    """stdlib round-robin: parallel fixed-output generations, aggregate t/s."""
    filler = ("The quick brown fox jumps over the lazy dog. " * (args.pp // 9 + 1))
    prompt = filler[:args.pp * 4] + "\n\nCount down from 500 to 1, one number per line."
    rows = []
    for conc in args.concurrency:
        tps = []
        for _ in range(args.runs):
            def one(_i):
                return chat(args.base_url, args.api_key, args.model,
                            [{"role": "user", "content": prompt}],
                            max_tokens=args.tg, timeout=args.timeout)
            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
                outs = list(ex.map(one, range(conc)))
            dt = time.monotonic() - t0
            gen = sum((o["usage"].get("completion_tokens") or 0) for o in outs)
            tps.append(gen / dt if dt > 0 else 0.0)
        agg = sum(tps) / len(tps)
        rows.append({"concurrency": conc, "tg_tps_agg": round(agg, 2),
                     "tg_tps_req": round(agg / conc, 2)})
        g = conc_gate(rows, args.concurrency)
        if g and not g[0]:
            print(f"NO CONCURRENCY: {g[1]} -- aborting bench early", flush=True)
            break
    return rows


def main():
    p = make_arg_parser("speed benchmark")
    p.add_argument("--pp", type=int, default=2048)
    p.add_argument("--tg", type=int, default=512)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--benchy-timeout", type=float, default=7200)
    p.add_argument("--fallback", action="store_true",
                   help="skip llama-benchy, use stdlib round-robin")
    args = p.parse_args()
    r = Result("speed")
    args.model = resolve_model(args.base_url, args.api_key, args.model)

    rows = None
    source = "fallback"
    tmp_json = os.path.join(os.path.dirname(args.save_result) if args.save_result else os.path.expanduser("~/.hermes/cache/scratch"), "benchy.tmp.json")
    if not args.fallback:
        try:
            rows = []
            for conc in sorted(set(args.concurrency)):
                data = run_benchy(args, tmp_json, [conc])
                if data is None:
                    rows = None
                    break
                rows += benchy_matrix(data)
                g = conc_gate(rows, args.concurrency)
                if g and not g[0]:
                    print(f"NO CONCURRENCY: {g[1]} -- aborting bench early",
                          flush=True)
                    break
            if rows:
                source = "llama-benchy"
        except Exception as e:  # noqa: BLE001
            print("llama-benchy failed, falling back: " + repr(e)[:300])
            rows = None
    if rows is None:
        rows = fallback_bench(args)

    for row in rows:
        agg = row.get("tg_tps_agg") or 0
        r.add(f"tg-c{row['concurrency']}", agg > 0,
              f"agg={agg:.1f} t/s req={row.get('tg_tps_req') or 0:.1f} "
              f"pp={row.get('pp_tps') or 'n/a'} ttfr={row.get('ttfr_s') or 'n/a'}")

    g = conc_gate(rows, args.concurrency)
    if g is not None:
        r.add("concurrency-scaling", g[0],
              g[1] + (" [bench aborted early]" if len(rows)
                      < len(set(args.concurrency)) else ""))

    print("\nsource: " + source)
    for row in rows:
        print("  " + json.dumps(row))
    r.finish(args, extra={"source": source, "matrix": rows})


if __name__ == "__main__":
    main()
