#!/usr/bin/env python3
"""Run the full qualification suite in order and print the verdict table.

Order follows the qualification gate:
  1 test_endpoint   (is it even up + routed right)
  2 test_coherence  (deterministic quality, temp 0)
  3 test_tool_leak  (no tool JSON / prompt echo leak; tool round-trip)
  4 test_multiturn  (agentic continuity across turns)
  5 test_needles    (retrieval at 3/4 of served context)
  6 test_speed      (pp2048/tg512 @ conc 1,2,4 — run LAST, zero interference)

Any FAIL = deployment NOT verified. Results JSONs land in --out-dir.

Usage:
  python3 run_all.py --model ds4f-gb10 --ctx 1048576
  python3 run_all.py --model ds4f-gb10 --skip speed        # quality only
"""
import argparse
import json
import os
import subprocess
import sys
import time

STEPS = [
    ("endpoint", "test_endpoint.py", []),
    ("coherence", "test_coherence.py", []),
    ("tool-leak", "test_tool_leak.py", []),
    ("multiturn", "test_multiturn.py", []),
    ("needles", "test_needles.py", ["--ctx", "{ctx}"]),
    ("speed", "test_speed.py", []),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.environ.get("QC_BASE_URL", "http://localhost:4000/v1"))
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("QC_API_KEY", ""))
    p.add_argument("--ctx", type=int, default=1048576,
                   help="served context length for the needle depth math")
    p.add_argument("--out-dir", default="results/" + time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--skip", nargs="*", default=[],
                   help="step names to skip: " + " ".join(s[0] for s in STEPS))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    verdicts = {}
    for name, script, extra in STEPS:
        if name in args.skip:
            print(f"== {name}: SKIPPED")
            continue
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), script),
               "--base-url", args.base_url, "--model", args.model,
               "--api-key", args.api_key,
               "--save-result", os.path.join(args.out_dir, name + ".json")]
        for e in extra:
            cmd.append(e.replace("{ctx}", str(args.ctx)))
        print(f"\n===== {name} =====\n$ " + " ".join(cmd), flush=True)
        t0 = time.monotonic()
        rc = subprocess.call(cmd)
        verdicts[name] = "PASS" if rc == 0 else "FAIL"
        print(f"===== {name}: {verdicts[name]} ({time.monotonic()-t0:.0f}s)", flush=True)

    print("\n--- qualification verdict ---")
    for k, v in verdicts.items():
        print(f"  {k:12s} {v}")
    ok = all(v == "PASS" for v in verdicts.values())
    print(f"\n{'VERIFIED' if ok else 'NOT VERIFIED'} — {args.model}")
    with open(os.path.join(args.out_dir, "verdict.json"), "w") as f:
        json.dump({"model": args.model, "base_url": args.base_url,
                   "verdicts": verdicts, "verified": ok}, f, indent=2)
    print("results: " + args.out_dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
