#!/usr/bin/env python3
"""Check: prefix caching must actually work on the engine.

Design (ungamable + no false fails):
  - one long unique prompt (fresh per-run nonce) sent 3x, temp 0
  - req1 is cold; req2/3 must show cache reuse
  - PASS needs: real prefill happened (req1 ttft >= 2s), req2 TTFT <
    req1/2, and cached-token evidence.
  - cached-token evidence: if the engine reports
    usage.prompt_tokens_details.cached_tokens, req2 must show >= 50%
    of prompt_tokens cached AND req1 must NOT already be "fully cached"
    (proves the nonce cold-start and that the counter isn't fabricated).
    If details is null/absent (some vLLM parsers do this — DS4F,
    2026-10 note), the counter gate is skipped and the TTFT delta
    carries the verdict (INFO line records the gap).
  - engines with NO prefix cache: TTFT does not drop => FAIL.
  - thinking stays engine-default ON (user rule 2026-11-24); output is
    capped small because we grade usage, not content.

Usage: python3 test_prefix_cache.py [--base-url URL] [--model ID] [--tokens 4096]
"""
import random
import time
import uuid

from qc_common import make_arg_parser, resolve_model, openai_base, Result
import json
import urllib.request


def call_stream(base_url, api_key, model, messages, max_tokens, timeout):
    """One streaming chat completion; returns (ttft_s, usage_dict)."""
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        openai_base(base_url) + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "***" + api_key} if api_key else {})})
    t0 = time.perf_counter()
    ttft = None
    usage = None
    for raw in urllib.request.urlopen(req, timeout=timeout):
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload_s = line[6:]
        if payload_s == "[DONE]":
            break
        try:
            d = json.loads(payload_s)
        except json.JSONDecodeError:
            continue
        if ttft is None:
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                ttft = time.perf_counter() - t0
        if d.get("usage"):
            usage = d["usage"]
    if ttft is None:
        ttft = time.perf_counter() - t0
    return ttft, usage or {}


def main():
    ap = make_arg_parser(__doc__)
    ap.add_argument("--tokens", type=int, default=4096,
                    help="filler words in the probe prompt (~2 tokens/word; default 4096)")
    args = ap.parse_args()
    args.model = resolve_model(args.base_url, args.api_key, args.model, timeout=30.0)

    rnd = random.Random(20261125)          # pinned generation
    nonce = uuid.uuid4().hex               # fresh per run -> req1 is cold
    filler = " ".join(f"w{rnd.randrange(10**9)}n{rnd.randrange(10**9)}"
                      for _ in range(args.tokens))
    msgs = [{"role": "user", "content":
             f"Nonce {nonce}. Memorize this sequence of tokens, you will be asked "
             f"about it later. SEQUENCE: {filler} END. How many tokens were there?"}]

    r = Result("prefix-cache")
    try:
        runs = []
        for _ in range(3):
            ttft, usage = call_stream(args.base_url, args.api_key, args.model,
                                      msgs, max_tokens=256, timeout=args.timeout)
            det = usage.get("prompt_tokens_details") or {}
            runs.append({"ttft": ttft,
                         "prompt_tokens": usage.get("prompt_tokens"),
                         "cached": det.get("cached_tokens"),
                         "details_present": "cached_tokens" in det})
            print(f"  run: ttft={runs[-1]['ttft']:.2f}s prompt={runs[-1]['prompt_tokens']} "
                  f"cached={runs[-1]['cached']}")
    except Exception as e:  # noqa: BLE001
        r.add("probe-requests", False, f"{type(e).__name__}: {e}")
        r.finish(args)
        return

    r1, r2 = runs[0], runs[1]
    pt = r2["prompt_tokens"] or 0
    r.add("usage-reported", pt > 0, f"prompt_tokens={pt}")
    r.add("cold-prefill", r1["ttft"] >= 2.0,
          f"req1 ttft={r1['ttft']:.2f}s — prompt too short to measure the gate"
          if r1["ttft"] < 2.0 else f"req1 ttft={r1['ttft']:.2f}s")
    r.add("warm-ttft-delta", r2["ttft"] < r1["ttft"] / 2,
          f"ttft1={r1['ttft']:.2f}s ttft2={r2['ttft']:.2f}s (need >2x drop)")

    if r2["details_present"]:
        c1 = r1["cached"] if r1["cached"] is not None else 0
        c2 = r2["cached"] or 0
        r.add("cold-not-faked", c1 <= 0.1 * max(pt, 1),
              f"req1 cached={c1} of {pt} (nonce prompt must not hit cache)")
        r.add("cached-tokens-hit", c2 >= 0.5 * pt,
              f"req2 cached={c2} of {pt} (need >=50%)")
    else:
        r.add("cached-tokens-field", True,
              "INFO: usage.prompt_tokens_details absent — TTFT delta carries the verdict")

    r.finish(args, extra={"runs": runs})


if __name__ == "__main__":
    main()
