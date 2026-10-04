#!/usr/bin/env python3
"""Check 2: needle-in-haystack retrieval at 3/4 of context length.

Generates N random unique needles (unique rare-ish words + random 5-digit
numbers, no shared tokens between needles, pinned seed) embedded at random
depths inside filler text sized to reach 3/4 of the model's context window,
then requires exact retrieval of each needle's number.

The bar is ungamable by pattern completion: every needle has a unique word
AND a unique random number; retrieval is checked per needle.

Client-side design (fixed 2026-09-13 — the old version burned more wall
time in the harness than the GPU did):
  * streaming (qc_stream) with an idle-stall detector, not a single huge
    non-streaming round trip — long generations survived only until some
    socket hiccup dropped them, then the whole call restarted from zero.
  * multi-turn continuation with per-turn output cap, so one 32K wall is
    never required; progress across turns is cumulative.
  * O(1)-per-line grading: the answer is tokenized ONCE per turn and every
    needle match is harvested in a single pass (the old grader ran 2048
    sequential re.search scans over an ever-growing answer per regrade —
    quadratic in total output).
  * unique-entry scoring: repeats the model loves do not count twice;
    misses are reported by needle.
  * stall detection: MAX_STALL_TURNS consecutive turns with no new needles
    ends the run (with its score) instead of letting the model loop.
"""
import argparse
import json
import random
import re
import sys
import time

from qc_common import make_arg_parser, resolve_model, Result
from qc_stream import stream_chat

WORDS = [
    "quartz", "falcon", "cedar", "velvet", "anvil", "tundra", "lantern",
    "harbor", "pebble", "cobalt", "meadow", "thistle", "glacier", "ember",
    "willow", "basalt", "saffron", "lagoon", "cinder", "poplar", "quiver",
    "marmot", "jasper", "opal", "birch", "garnet", "prairie", "summit",
    "sable", "tundra", "walnut", "onyx", "kettle", "marble", "pepper",
    "timber", "beetle", "corvid", "gypsum", "ferns",
]


def _word_tag(k):
    """Bijective base-26 tag: a, b, ... z, aa, ab, ... (alphabetic so it
    matches the LINE_RE word grammar and lets us mint >200 unique words)."""
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(97 + r) + s
    return s


def gen_needles(rng, n):
    """n unique (word, number) pairs.

    Words are minted as '<base>-<suffix>-<tag>' so we can produce ARBITRARILY
    many unique identifiers (the old version only had ~200 reachable words
    and looped forever when asked for 2048 unique needles). Numbers are
    unique 5-digit values. Both are pinned by the caller's seeded RNG.
    """
    base = list(dict.fromkeys(WORDS))            # dedupe, keep order (39)
    suff = ["stone", "brook", "haven", "ridge", "vale"]
    nb, ns = len(base), len(suff)
    seen_nums = set()
    needles = []
    for i in range(n):
        # (base_index, suff_index, tag) uniquely determines i -> unique word
        w = f"{base[i % nb]}-{suff[(i // nb) % ns]}-{_word_tag(i // (nb * ns))}"
        while True:
            num = rng.randrange(10000, 100000)
            if num not in seen_nums:
                seen_nums.add(num)
                break
        needles.append((w, num))
    assert len(set(w for w, _ in needles)) == n, "needle words must be unique"
    assert len(set(k for _, k in needles)) == n, "needle numbers must be unique"
    return needles


def filler_paragraph(rng, idx):
    animals = ["otter", "heron", "badger", "sparrow", "tortoise", "mole",
               "finch", "newt", "ibis", "lemur"]
    return (f"Section {idx}. The {rng.choice(animals)} crossed the quiet valley "
            f"at dawn while the {rng.choice(animals)} watched from the ridge. "
            "Nothing remarkable happened that day, and the records list no "
            "further events of note in this passage of the archive. ") * 4


def calibrate_ratio(base_url, api_key, model, sample, timeout=120):
    """Measure real chars/token on THIS endpoint with a throwaway request.

    The sample must be the needle-bearing prompt mix (needle lines are denser
    than filler), else the haystack is sized on filler-only chars/token and
    the real prompt overshoots 3/4 ctx -> HTTP 400. Calibrate on the mixed
    content actually sent.
    """
    probe = sample[:50000]
    from qc_common import chat
    out = chat(base_url, api_key, model,
               [{"role": "user", "content": "Reply OK. " + probe}],
               max_tokens=8, timeout=timeout)
    pt = out["usage"].get("prompt_tokens") or 0
    if pt < 50:
        raise RuntimeError(f"calibration probe got prompt_tokens={pt}")
    return len(probe) / pt


def build_haystack(rng, depth, n_needles, approx_tokens, cpt=4.0):
    needles = gen_needles(rng, n_needles)
    total_chars = int(approx_tokens * cpt)
    needle_lines = [f"The secret codeword for {w} is the number {k}." for w, k in needles]
    needle_chars = sum(len(x) for x in needle_lines)
    filler_chars = total_chars - needle_chars
    filler_para = filler_paragraph(rng, -1)
    para_chars = len(filler_para)
    # n_paras from the filler budget (NOT floored to needle count — at 2048
    # needles a one-needle-per-para floor forces each para to also carry a
    # fixed 852-char filler and the haystack overshoots 3/4 ctx -> 400). Each
    # para holds multiple needles via the uniform embed_at distribution.
    n_paras = max(4, int(filler_chars // para_chars * 0.97))

    # EVEN distribution across the whole haystack (user directive 2026-09-13):
    # one needle per stratum, jittered inside its stratum (pinned by caller rng).
    embed_at = sorted((i + rng.random()) / n_needles for i in range(n_needles))
    del depth  # kept in CLI for compat; distribution is now full-context even
    paras = []
    ni = 0
    for p in range(n_paras):
        frac = p / n_paras
        while ni < n_needles and frac >= embed_at[ni]:
            paras.append(needle_lines[ni])
            ni += 1
        paras.append(filler_para)
    while ni < n_needles:
        paras.append(needle_lines[ni])
        ni += 1
    return "\n\n".join(paras), needles


# --- single-pass grader -------------------------------------------------------
LINE_RE = re.compile(r"(?<![-\w])([A-Za-z]+(?:-[A-Za-z]+)*):\s*(\d{5})")


def harvest(answer_text, needle_map):
    """One pass over the answer: {word: number} for lines that match a planted
    needle EXACTLY (word known and number correct). Returns (hits:set, dup:int)."""
    hits = set()
    dup = 0
    for m in LINE_RE.finditer(answer_text):
        w, num = m.group(1), int(m.group(2))
        exp = needle_map.get(w)
        if exp is not None:
            if num == exp:
                if w in hits:
                    dup += 1
                hits.add(w)
    return hits, dup


def main():
    p = make_arg_parser("needle retrieval at depth")
    p.add_argument("--needles", type=int, default=64, help="count (default 64)")
    p.add_argument("--ctx", type=int, default=None,
                   help="model context length in tokens (the bar is 3/4 of served context)")
    p.add_argument("--depth", type=float, default=0.75,
                   help="fraction of context where needles sit (default 0.75)")
    p.add_argument("--seed", type=int, default=1234, help="pinned RNG seed")
    p.add_argument("--turn-tokens", type=int, default=None,
                   help="max_tokens per continuation turn (default: 1/4 of "
                        "full served context, per the qualification bar)")
    p.add_argument("--max-turns", type=int, default=10,
                   help="hard cap on continuation turns (default 10)")
    p.add_argument("--max-stall-turns", type=int, default=3,
                   help="end the run after N consecutive zero-progress turns (default 3)")
    p.add_argument("--no-thinking", action="store_true",
                   help="send chat_template_kwargs enable_thinking=false — recall probes: "
                        "keeps the entire output budget for the enumeration (this vLLM build "
                        "never streams reasoning deltas anyway, so thinking-ON burns window "
                        "tokens invisibly)")
    p.add_argument("--idle-stall-s", type=float, default=300.0,
                   help="fail a turn if no SSE chunk arrives within this many seconds")
    args = p.parse_args()
    r = Result("needles")
    args.model = resolve_model(args.base_url, args.api_key, args.model)

    ctx = args.ctx or 32768
    # output budget per turn: default 1/4 of FULL served context (user bar)
    turn_tokens = args.turn_tokens or (ctx // 4)
    rng = random.Random(args.seed)
    target_tokens = int(ctx * args.depth)
    # Two-pass calibration: build the haystack once at a guess, measure the
    # REAL chars/token on an actual slice (the needle lines are letter-denser
    # than filler, so a pure-filler cpt undersizes the haystack and the real
    # prompt overshoots 3/4 ctx -> HTTP 400), then rebuild at the true cpt.
    # The needle:filler ratio depends on --needles, so only a measured slice
    # reflects it.
    cpt = None
    # Iterative exact calibration (fixed point): build the haystack at a cpt,
    # form the FULL question, measure its real prompt_tokens with a cheap
    # non-stream request (max_tokens=8), and correct cpt toward the 3/4 target.
    # Rebuilding changes the needle:filler ratio (n_paras depends on cpt), so
    # one shot does not land — iterate until |measured - target| is small. This
    # replaces the slice-based estimate, which was not representative of the
    # 2048-needle mix and the real prompt overshooted 3/4 ctx -> HTTP 400.
    q = None
    needles = None
    for _iter in range(6):
        _rng = random.Random(args.seed)
        _hay, _needles = build_haystack(_rng, args.depth, args.needles, target_tokens, cpt=cpt or 4.3)
        _q = ("Below is a long archive. Follow ONLY the final instruction. "
              "For each of these codewords, state its secret number as "
              "'<word>: <number>' one per line, nothing else.\nCodewords: "
              + ", ".join(w for w, _ in _needles) + "\n\nARCHIVE:\n" + _hay)
        try:
            from qc_common import chat
            out = chat(args.base_url, args.api_key, args.model,
                       [{"role": "user", "content": _q}], max_tokens=8,
                       timeout=args.timeout)
            pt = (out.get("usage") or {}).get("prompt_tokens") or 0
            if pt <= 50:
                raise RuntimeError(f"probe prompt_tokens={pt}")
            new_cpt = len(_q) / pt
            print(f"iter {_iter}: cpt={new_cpt:.3f} prompt_tokens={pt} target={target_tokens}", flush=True)
            if abs(pt - target_tokens) / target_tokens < 0.02:
                cpt = new_cpt; q = _q; hay = _hay; needles = _needles
                break
            cpt = new_cpt
            q, hay, needles = _q, _hay, _needles
        except Exception as e:   # noqa: BLE001
            print("WARN exact cpt calibration failed: " + repr(e)[:200])
            cpt = 4.0
    if hay is None or q is None or needles is None:
        # Every calibration probe failed (engine busy/dead). Abort loudly —
        # grading a fallback-4.3-cpt haystack against a wrong-size prompt
        # produced bogus runs, and before this guard the test crashed with
        # UnboundLocalError instead.
        raise SystemExit("FATAL: cpt calibration never succeeded (all probes failed) — aborting run")
    hay = hay
    needle_map = dict(needles)
    n = len(needles)
    print(f"haystack: {len(hay)} chars @ {cpt:.2f} c/t -> "
          f"~{int(len(hay)/cpt)} tokens, {n} needles @ depth {args.depth} of ctx {ctx}",
          flush=True)

    question = q

    messages = [{"role": "user", "content": question}]
    full_answer = []          # pieces of assistant text across turns
    hits = set()
    started = time.monotonic()

    # 2026-09-29 fix: use EXACT token accounting from the previous request's
    # usage (char/cpt estimates badly undercount digit-heavy output text — the
    # list lines are numbers, ~2 chars/token). Without this, a continuation
    # request can overrun ctx and the engine answers 400 ("Input length ...
    # exceeds model's maximum context length"), killing the run ungraded.
    nudge_text = ("Continue the list exactly where you stopped. "
                  "Output ONLY remaining '<word>: <number>' lines, "
                  "one per line, nothing else.")
    prev_in = None            # exact prompt_tokens of the previous request
    prev_out = 0              # exact completion_tokens of the previous turn
    _ctk = ({"enable_thinking": False} if args.no_thinking
            else {"enable_thinking": True, "preserve_thinking": True})
    estimated_input = (sum(len(str(m["content"])) for m in messages)
                       + sum(len(s) for s in full_answer)) / cpt
    for turn in range(1, args.max_turns + 1):
        # fit check: input + output budget must fit the window
        if prev_in is not None:
            estimated_input = prev_in + prev_out + len(nudge_text) / cpt + 64
        cap = min(turn_tokens, int(ctx - estimated_input - 256))
        if cap < 2048:
            print(f"turn {turn}: no room left in ctx (input ~{int(estimated_input)}, "
                  f"cap {cap}) — ending chain", flush=True)
            break
        t0 = time.monotonic()
        try:
            out = stream_chat(args.base_url, args.api_key, args.model, messages,
                              max_tokens=cap, temperature=0.0,
                              idle_stall_s=args.idle_stall_s,
                              chat_template_kwargs=_ctk)
        except Exception as e:   # noqa: BLE001
            print(f"turn {turn}: request failed ({repr(e)[:400]}) — ending chain",
                  flush=True)
            break
        ans = (out.get("content") or "") + "\n" + (out.get("reasoning") or "")
        full_answer.append(ans)
        new_hits, dup = harvest(ans, needle_map)
        gained = new_hits - hits
        hits |= new_hits
        el = time.monotonic() - t0
        usage = out.get("usage") or {}
        prev_in = usage.get("prompt_tokens")
        prev_out = usage.get("completion_tokens") or 0
        reas = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        print(f"turn {turn}: +{len(gained)} new unique={len(hits)}/{n} "
              f"(dup {dup}) in={prev_in} out={prev_out} reas={reas} "
              f"cached={ (usage.get('prompt_tokens_details') or {}).get('cached_tokens') } "
              f"finish={out.get('finish')} {el:.0f}s "
              f"[total {time.monotonic()-started:.0f}s]", flush=True)
        messages.append({"role": "assistant", "content": ans})
        if len(hits) == n:
            break
        if not gained:
            if turn - getattr(main, "_last_gain_turn", 0) >= args.max_stall_turns + 1 and \
               getattr(main, "_stall_count", 0) >= args.max_stall_turns:
                print(f"stalled: {args.max_stall_turns} zero-progress turns — ending", flush=True)
                break
            main._stall_count = getattr(main, "_stall_count", 0) + 1
        else:
            main._stall_count = 0
            main._last_gain_turn = turn
        # nudge to continue after a self-restart signature (skill: model ends
        # with "let me go through it again..." then EOS)
        messages.append({"role": "user", "content": nudge_text})

    ans_all = "\n".join(full_answer)
    misses = [w for w, _ in needles if w not in hits]
    ok = len(hits) == n
    r.add(f"recall-{len(hits)}/{n}", ok,
          f"misses={misses[:8]} ({len(misses)} total) elapsed={time.monotonic()-started:.0f}s")
    if args.save_result:
        with open(args.save_result + ".answer.txt", "w") as f:
            f.write(ans_all)
        with open(args.save_result + ".needles.json", "w") as f:
            json.dump([[w, k] for w, k in needles], f)

    r.finish(args, extra={"needles": n, "ctx": ctx, "depth": args.depth,
                          "seed": args.seed, "turns": len(full_answer),
                          "elapsed_s": round(time.monotonic() - started, 1)})


if __name__ == "__main__":
    main()
