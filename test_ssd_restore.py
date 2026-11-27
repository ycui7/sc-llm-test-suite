#!/usr/bin/env python3
"""SSD restore coherence gate (user directive 2026-10-19).

History: on GB10 unified-memory boxes, KV restored from SSD came back as
garbage (LMCache forum blocker #2 — DSA dual-group restore divergence).
This gate makes 'loaded from SSD' PROVABLE and its output CORRECT:

  cold phase : long story+needles prompt @ temp 0 -> answer O1 (KV computed)
  (engine restart empties the GPU KV pool; the FS tier on NVMe survives)
  warm phase : same prompt -> answer O2 must come from disk restore.
      gates:  (a) speedup = ttft_cold/ttft_warm >= 10  (restore vs recompute)
              (b) O2 == O1 normalized (temp-0 determinism; garbage/divergence
                  = the known SSD bug -> FAIL, never ship silently)
              (c) needle recall exact on the warm answer (harvest)

Harness order: run --phase cold, restart engine, run --phase warm --expect <cold.json>.
"""
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_common import (make_arg_parser, chat, resolve_model, Result)  # noqa: E402
from test_needles import gen_needles, load_story, harvest, LINE_RE  # noqa: E402
from test_needles import build_haystack  # noqa: E402

CPT = 5.0  # chars/token calibration used by the suite


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def compose(hay, needles):
    return ("Below is a long story archive. Follow ONLY the final instruction. "
            "For each of these codewords, state its secret number as "
            "'<word>: <number>' one per line, nothing else.\nCodewords: "
            + ", ".join(w for w, _ in needles) + "\n\nARCHIVE:\n" + hay)


def main():
    p = make_arg_parser("ssd restore coherence gate")
    p.add_argument("--tokens", type=int, default=48000)
    p.add_argument("--needles", type=int, default=24)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--phase", choices=["cold", "warm"], default="cold")
    p.add_argument("--expect", default=None, help="cold json (warm phase)")
    p.add_argument("--min-speedup", type=float, default=10.0)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    args.model = resolve_model(args.base_url, args.api_key, args.model)
    rng = random.Random(args.seed)
    story_paras = load_story()
    hay, needles = build_haystack(rng, args.needles, int(args.tokens * CPT),
                                  story_paras)
    q = compose(hay, needles)
    needle_map = dict(needles)

    out = chat(args.base_url, args.api_key, args.model,
               [{"role": "user", "content": q}],
               max_tokens=args.max_tokens, timeout=3600.0, temperature=0.0)
    ans = out["content"]
    ttft = out["ttft_s"]
    hits, dup = harvest(ans, needle_map)

    r = Result("ssd-restore-" + args.phase)
    path = args.save or f"ssd_restore_{args.phase}_{args.seed}.json"

    if args.phase == "cold":
        r.add("cold-needles", len(hits) == len(needles),
              f"{len(hits)}/{len(needles)} (computed KV must already be right)")
        r.add("cold-answered", len(norm(ans)) > 30,
              f"ttft={ttft:.0f}s finish={out['finish']}")
        json.dump({"answer": ans, "ttft": ttft, "seed": args.seed,
                   "tokens": args.tokens, "needles": args.needles},
                  open(path, "w"))
    else:
        assert args.expect, "warm phase needs --expect <cold.json>"
        cold = json.load(open(args.expect))
        speedup = cold["ttft"] / max(ttft, 0.01)
        r.add("restore-from-disk", speedup >= args.min_speedup,
              f"speedup={speedup:.1f}x cold={cold['ttft']:.0f}s warm={ttft:.0f}s "
              f"(GPU KV emptied by restart; tier survived on NVMe)")
        same = norm(ans) == norm(cold["answer"])
        r.add("restore-coherent(temp0-identical)", same,
              "answer identical to cold" if same else
              "DIVERGED: restored KV != computed KV — the SSD-garbage bug\n"
              "cold[:200]=" + cold["answer"][:200] + "\nwarm[:200]=" + ans[:200])
        r.add("warm-needles", len(hits) == len(needles),
              f"{len(hits)}/{len(needles)} dup={dup}")
        json.dump({"answer": ans, "ttft": ttft, "speedup": speedup,
                   "identical": same}, open(path, "w"))

    r.finish(args=args, extra={"phase": args.phase, "seed": args.seed})
    sys.exit(0 if r.ok else 1)


if __name__ == "__main__":
    main()
