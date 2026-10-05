#!/usr/bin/env python3
"""Check 2: needle-in-haystack retrieval — story haystack, single inference.

Spec (user directive 2026-10-05, supersedes the 3/4-window random-filler
design of 2026-09):
  1. Needles UNIFORMLY distributed: one per stratum of the needle region
     (stratified-jitter, pinned seed -> reproducible AND uniform by
     construction, unlike pure-random which clumps).
  2. Background is a COHERENT STORY: real public-domain novel prose from
     corpus/ (Project Gutenberg; fetched by fetch_corpus.sh). Never random
     filler. --story off keeps the legacy filler mode for A/B only.
  3. Needles distributed from the start up to min(2/3 of max context, 512K)
     tokens. (For ctx 1M -> 512K; for ctx 64K -> 42.7K.)
  4. Output room is RESERVED: max_tokens = min(n*PER_LINE_ALLOW, ctx -
     measured_prompt - MARGIN). If the reserved budget cannot fit n lines at
     all, the run ABORTS LOUDLY (never silently clips the answer space).
  5. ALL needles must come back in ONE single LLM inference (max-turns
     default 1 — continuation turns are the old escape hatch and now
     require explicit --max-turns >1). Grading is on that one response.
  6. Needle count scales with context: ctx <= SHORT_CTX_MAX (64K) uses 1024
     needles; otherwise 2048. Explicit --needles still overrides.

Ungamable design retained: every needle is a unique minted word AND a
unique random 5-digit number; per-needle exact-match grading; pinned seeds.
Streaming with idle-stall detection retained (qc_stream).
"""
import argparse
import json
import os
import random
import re
import sys
import time

from qc_common import make_arg_parser, resolve_model, Result
from qc_stream import stream_chat

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
# concatenation order for the story haystack
CORPUS_FILES = ["pride_and_prejudice.txt", "sherlock_holmes.txt",
                "great_expectations.txt", "moby_dick.txt"]

NEEDLE_DEPTH_FRAC = 2.0 / 3.0     # rule 3
NEEDLE_DEPTH_CAP = 512_000        # rule 3
SHORT_CTX_MAX = 65_536            # rule 6
PER_LINE_ALLOW = 40               # tokens budgeted per needle line (rule 4;
                                  # observed 12-25 tok/line incl. self-restart)
OUT_MARGIN = 512                  # tokens kept free above ctx boundary (rule 4)

WORDS = [
    "quartz", "falcon", "cedar", "velvet", "anvil", "tundra", "lantern",
    "harbor", "pebble", "cobalt", "meadow", "thistle", "glacier", "ember",
    "willow", "basalt", "saffron", "lagoon", "cinder", "poplar", "quiver",
    "marmot", "jasper", "opal", "birch", "garnet", "prairie", "summit",
    "sable", "walnut", "onyx", "kettle", "marble", "pepper",
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
    """n unique (word, number) pairs — minted words, unique 5-digit numbers."""
    base = list(dict.fromkeys(WORDS))
    suff = ["stone", "brook", "haven", "ridge", "vale"]
    nb, ns = len(base), len(suff)
    seen_nums = set()
    needles = []
    for i in range(n):
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


def load_story(corpus_dir=CORPUS_DIR, files=CORPUS_FILES):
    """Coherent novel prose as paragraphs (rule 2). Strips the Gutenberg
    header/footer; keeps only substantial paragraphs (>=200 chars) so TOC /
    advertisement cruft never enters the haystack."""
    paras = []
    for fn in files:
        p = os.path.join(corpus_dir, fn)
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: corpus file missing: {p} — run fetch_corpus.sh")
        raw = open(p, encoding="utf-8", errors="replace").read()
        m = re.search(r"\*\*\*\s*START OF (THIS |THE )?PROJECT GUTENBERG.*", raw)
        if m:
            raw = raw[m.end():]
        m = re.search(r"\*\*\*\s*END OF (THIS |THE )?PROJECT GUTENBERG.*", raw)
        if m:
            raw = raw[:m.start()]
        ps = [x.strip().replace("\n", " ") for x in re.split(r"\n\s*\n", raw)]
        paras.extend(x for x in ps if len(x) >= 200)
    total = sum(len(p) for p in paras)
    print(f"story corpus: {len(paras)} paragraphs, {total} chars "
          f"(~{int(total/5.0)} tokens est) from {', '.join(files)}", flush=True)
    return paras


def filler_paragraph(rng, idx):
    animals = ["otter", "heron", "badger", "sparrow", "tortoise", "mole",
               "finch", "newt", "ibis", "lemur"]
    return (f"Section {idx}. The {rng.choice(animals)} crossed the quiet valley "
            f"at dawn while the {rng.choice(animals)} watched from the ridge. "
            "Nothing remarkable happened that day, and the records list no "
            "further events of note in this passage of the archive. ") * 4


def build_haystack(rng, n_needles, body_chars, story_paras):
    """Story haystack with UNIFORMLY distributed needles (rules 1+2+3).

    Walk story paragraphs to fill body_chars; plant needle i as its own
    sentence once the walk crosses that needle's stratum boundary
    ((i + jitter)/n) — uniform by construction, reproducible under a pinned
    seed. Needles span the whole (needle-region-sized) body, i.e. from the
    start up to the rule-3 boundary token count.
    """
    needles = gen_needles(rng, n_needles)
    needle_lines = [f"The secret codeword for {w} is the number {k}."
                    for w, k in needles]
    nl_chars = sum(len(x) for x in needle_lines)
    body_budget = max(1000, body_chars - nl_chars)
    embed_at = [(i + rng.random()) / n_needles for i in range(n_needles)]

    out = []
    chars = 0
    ni = 0
    pi = 0
    n_paras = len(story_paras)
    guard = 0
    while ni < n_needles and chars < body_budget:
        frac = chars / body_budget
        while ni < n_needles and frac >= embed_at[ni]:
            out.append(needle_lines[ni])
            chars += len(needle_lines[ni]) + 2
            ni += 1
        if ni >= n_needles or chars >= body_budget:
            break
        p = story_paras[pi % n_paras]
        pi += 1
        guard += 1
        if guard > 4_000_000:
            break
        out.append(p)
        chars += len(p) + 2
    # plant any remainder (pool/budget edge) evenly at the tail
    while ni < n_needles:
        out.append(needle_lines[ni])
        ni += 1
    hay = "\n\n".join(out)

    # uniformity receipt (rule 1 proof): fractional positions + gap stats
    nlset = set(needle_lines)
    pos = []
    run = 0
    for seg in out:
        if seg in nlset:
            pos.append(run / max(1, len(hay)))
        run += len(seg) + 2
    if len(pos) >= 2:
        gaps = [b - a for a, b in zip(pos, pos[1:])]
        gm = sum(gaps) / len(gaps)
        print(f"needle distribution: first={pos[0]:.4f} last={pos[-1]:.4f} "
              f"gap_mean={gm:.5f} gap_max={max(gaps):.5f} gap_min={min(gaps):.5f} "
              f"(uniform expected gap={1.0/len(pos):.5f})", flush=True)
    return hay, needles


def build_haystack_legacy(rng, n_needles, approx_tokens, cpt):
    """Pre-2026-10-05 random-filler mode (--story off, A/B only)."""
    needles = gen_needles(rng, n_needles)
    total_chars = int(approx_tokens * cpt)
    needle_lines = [f"The secret codeword for {w} is the number {k}."
                    for w, k in needles]
    needle_chars = sum(len(x) for x in needle_lines)
    filler_chars = total_chars - needle_chars
    filler_para = filler_paragraph(rng, -1)
    para_chars = len(filler_para)
    n_paras = max(4, int(filler_chars // para_chars * 0.97))
    embed_at = sorted((i + rng.random()) / n_needles for i in range(n_needles))
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
    """One pass over the answer: {word} for lines that match a planted needle
    EXACTLY (word known and number correct). Returns (hits:set, dup:int)."""
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
    p = make_arg_parser("needle retrieval — story haystack, single inference")
    p.add_argument("--needles", type=int, default=None,
                   help="count; default per rule 6: ctx<=64K -> 1024, else 2048")
    p.add_argument("--ctx", type=int, default=None,
                   help="model context length in tokens")
    p.add_argument("--depth", type=float, default=None,
                   help="override needle-region token target (default rule 3: "
                        "min(2/3*ctx, 512K))")
    p.add_argument("--seed", type=int, default=1234, help="pinned RNG seed")
    p.add_argument("--turn-tokens", type=int, default=None,
                   help="override output budget (default rule 4: "
                        "min(n*%d, ctx - prompt - %d))" % (PER_LINE_ALLOW, OUT_MARGIN))
    p.add_argument("--max-turns", type=int, default=1,
                   help="continuation turns beyond the required single "
                        "inference (rule 5: keep 1; >1 = legacy escape hatch)")
    p.add_argument("--max-stall-turns", type=int, default=3)
    p.add_argument("--no-thinking", action="store_true", default=True,
                   help="enable_thinking=false (default): recall probes keep "
                        "the whole output budget for the enumeration")
    p.add_argument("--thinking", dest="no_thinking", action="store_false",
                   help="re-enable reasoning tokens (not recommended)")
    p.add_argument("--story", choices=["auto", "off"], default="auto",
                   help="haystack background: auto = public-domain novels "
                        "(rule 2); off = legacy random filler (A/B only)")
    p.add_argument("--idle-stall-s", type=float, default=300.0,
                   help="fail a turn if no SSE chunk arrives within this many seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="build + report the haystack, send nothing")
    args = p.parse_args()
    r = Result("needles")
    args.model = resolve_model(args.base_url, args.api_key, args.model)

    ctx = args.ctx or 32768
    n_needles = args.needles or (1024 if ctx <= SHORT_CTX_MAX else 2048)  # r6
    target_tokens = int(args.depth * ctx) if args.depth is not None else \
        min(int(ctx * NEEDLE_DEPTH_FRAC), NEEDLE_DEPTH_CAP)               # r3
    rng = random.Random(args.seed)

    story_paras = load_story() if args.story == "auto" else None

    def compose(_hay, _needles):
        return ("Below is a long story archive. Follow ONLY the final "
                "instruction. For each of these codewords, state its secret "
                "number as '<word>: <number>' one per line, nothing else.\n"
                "Codewords: " + ", ".join(w for w, _ in _needles) +
                "\n\nARCHIVE:\n" + _hay)

    # Iterative exact calibration to the needle-region token target (the
    # fixed-point scheme from the 09-29 rewrite, kept).
    cpt = None
    q = hay = needles = None
    measured_pt = None
    all_probes_failed = True
    for _iter in range(6):
        _rng = random.Random(args.seed)
        if story_paras is not None:
            _hay, _needles = build_haystack(_rng, n_needles,
                                            int(target_tokens * (cpt or 5.0)),
                                            story_paras)
        else:
            _hay, _needles = build_haystack_legacy(_rng, n_needles,
                                                   target_tokens, cpt or 5.0)
        _q = compose(_hay, _needles)
        if args.dry_run:
            print(f"dry-run iter {_iter}: hay_chars={len(_hay)} "
                  f"q_chars={len(_q)}", flush=True)
            q, hay, needles, cpt = _q, _hay, _needles, (cpt or 5.0)
            break
        try:
            from qc_common import chat
            out = chat(args.base_url, args.api_key, args.model,
                       [{"role": "user", "content": _q}], max_tokens=8,
                       timeout=args.timeout)
            pt = (out.get("usage") or {}).get("prompt_tokens") or 0
            if pt <= 50:
                raise RuntimeError(f"probe prompt_tokens={pt}")
            new_cpt = len(_q) / pt
            print(f"iter {_iter}: cpt={new_cpt:.3f} prompt_tokens={pt} "
                  f"target={target_tokens}", flush=True)
            all_probes_failed = False
            cpt = new_cpt
            q, hay, needles, measured_pt = _q, _hay, _needles, pt
            if abs(pt - target_tokens) / target_tokens < 0.02:
                break
        except Exception as e:   # noqa: BLE001
            print("WARN exact cpt calibration failed: " + repr(e)[:200],
                  flush=True)
    if hay is None or q is None or needles is None:
        raise SystemExit("FATAL: haystack never built / all probes failed — "
                         "aborting run, nothing was graded")
    if cpt is None:
        cpt = 5.0
    if measured_pt is None:
        # probes never succeeded but the legacy builder produced a haystack;
        # fall back to the char estimate (loud note, not silent).
        print("WARN: no server-measured prompt_tokens; using char/cpt estimate",
              flush=True)
        measured_pt = int(len(q) / cpt)
    if args.dry_run:
        print(f"dry-run complete: {n_needles} needles, needle-region target "
              f"{target_tokens} tokens (rule 3), q_chars={len(q)} — no "
              f"request sent", flush=True)
        return
    needle_map = dict(needles)
    n = len(needles)
    print(f"haystack: {len(hay)} chars @ {cpt:.2f} c/t -> ~{int(len(hay)/cpt)} "
          f"tokens, {n} needles across needle region {target_tokens} tokens "
          f"(rule 3: min(2/3*ctx={int(ctx*2//3)}, {NEEDLE_DEPTH_CAP})), "
          f"story={args.story}", flush=True)

    messages = [{"role": "user", "content": q}]
    full_answer = []
    hits = set()
    started = time.monotonic()

    prev_in = measured_pt
    prev_out = 0
    _ctk = ({"enable_thinking": False} if args.no_thinking
            else {"enable_thinking": True, "preserve_thinking": True})
    nudge_text = ("Continue the list exactly where you stopped. "
                  "Output ONLY remaining '<word>: <number>' lines, "
                  "one per line, nothing else.")

    single_inference = (args.max_turns == 1)
    for turn in range(1, args.max_turns + 1):
        # rule 4: reserve real output room from the MEASURED prompt size
        est_in = float(prev_in or 0) + (prev_out or 0) \
            + (len(nudge_text) / cpt if turn > 1 else 0) \
            + (0 if turn == 1 else 64)
        room = int(ctx - est_in - OUT_MARGIN)
        budget = min(n * PER_LINE_ALLOW, room)
        if args.turn_tokens:
            budget = args.turn_tokens
        if budget < n * 14:
            raise SystemExit(
                f"FATAL (rule 4): output room {budget} tok < {n} needles x14 "
                f"(est prompt {int(est_in)}, ctx {ctx}, margin {OUT_MARGIN}). "
                f"Not enough space to return every needle in one inference — "
                f"aborting loudly; context is silently clipped NEVER.")
        if turn == 1:
            print(f"rule 4: est_prompt={int(est_in)} output_budget={budget} "
                  f"(n*{PER_LINE_ALLOW}={n*PER_LINE_ALLOW}, room={room})",
                  flush=True)
        t0 = time.monotonic()
        try:
            out = stream_chat(args.base_url, args.api_key, args.model,
                              messages, max_tokens=budget, temperature=0.0,
                              idle_stall_s=args.idle_stall_s,
                              chat_template_kwargs=_ctk)
        except Exception as e:   # noqa: BLE001
            print(f"turn {turn}: request failed ({repr(e)[:400]}) — ending "
                  f"chain", flush=True)
            break
        ans = (out.get("content") or "") + "\n" + (out.get("reasoning") or "")
        full_answer.append(ans)
        new_hits, dup = harvest(ans, needle_map)
        gained = new_hits - hits
        hits |= new_hits
        el = time.monotonic() - t0
        usage = out.get("usage") or {}
        prev_in = usage.get("prompt_tokens") or prev_in
        prev_out = usage.get("completion_tokens") or prev_out
        reas = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        print(f"turn {turn}: +{len(gained)} new unique={len(hits)}/{n} "
              f"(dup {dup}) in={prev_in} out={prev_out} reas={reas} "
              f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')} "
              f"finish={out.get('finish')} {el:.0f}s "
              f"[total {time.monotonic()-started:.0f}s]", flush=True)
        if single_inference:
            break                       # rule 5: one inference decides
        messages.append({"role": "assistant", "content": ans})
        if len(hits) == n:
            break
        if not gained:
            if getattr(main, "_stall_count", 0) >= args.max_stall_turns:
                print(f"stalled: {args.max_stall_turns} zero-progress turns — "
                      f"ending", flush=True)
                break
            main._stall_count = getattr(main, "_stall_count", 0) + 1
        else:
            main._stall_count = 0
        messages.append({"role": "user", "content": nudge_text})

    ans_all = "\n".join(full_answer)
    misses = [w for w, _ in needles if w not in hits]
    ok = len(hits) == n
    r.add(f"recall-{len(hits)}/{n}", ok,
          f"misses={misses[:8]} ({len(misses)} total) "
          f"single_inference={single_inference} "
          f"elapsed={time.monotonic()-started:.0f}s")
    if args.save_result:
        with open(args.save_result + ".answer.txt", "w") as f:
            f.write(ans_all)
        with open(args.save_result + ".needles.json", "w") as f:
            json.dump([[w, k] for w, k in needles], f)

    r.finish(args, extra={"needles": n, "ctx": ctx,
                          "needle_region_tokens": target_tokens,
                          "story": args.story, "seed": args.seed,
                          "turns": len(full_answer),
                          "single_inference": single_inference,
                          "elapsed_s": round(time.monotonic() - started, 1)})


if __name__ == "__main__":
    main()
