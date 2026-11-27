#!/usr/bin/env python3
"""qc_hist — box-drawn histogram of MISSED needles across the tested span.

Used by test_needles.py; standalone demo:  python3 qc_hist.py --demo

Layout rules (each forced by a real terminal failure):
  * NO braille: font fallback gave empty U+2800 and filled cells different
    advances -> rows drifted against each other.
  * ONE glyph family inside the box: ASCII + half/three-quarter blocks
    (U+2580-U+258F) + SINGLE-line box drawing. Double-line (U+2550+) is
    banned by _check().
  * EVERY framed row is produced by one function, _row(), with a single
    fixed gutter width (Y-label field) reserved on *every* line.
  * The x-axis token labels live INSIDE the box (between the lower border
    and the bottom corner rule), stamped at the exact plot column of the
    tick they belong to. Nothing aligns to the box from outside anymore.
  * Sub-cell packing: 2 logical rows per text line via half-blocks: the
    32-needle scale in 16 lines. 1 dot = 1 missed needle, ceiling 32.
  * x SPAN: the qualification bar is needles-at-3/4-context, so the axis
    covers 3/4 of the served context (0 .. ctx*3/4), divided into 64
    equal bins. Full served context is NOT shown.
"""
import argparse
import random

BINS = 64
ROWS = 32
FULL = 32
PACK = 2
DEPTH = 0.75                                # tested span fraction of ctx
GUT = len(str(ROWS)) + 1                      # Y-label field, fixed for ALL

BLOCK = {0: " ", 1: "\u2580", 2: "\u2584", 3: "\u2588"}
ALLOWED = (set(range(32, 127))
           | {0x2500, 0x2502, 0x250C, 0x2510, 0x2514, 0x2518,
              0x251C, 0x2524, 0x252C, 0x2534, 0x253C}
           | {0x2580, 0x2584, 0x2588})
BOX_TOP, BOX_TOPR = "\u250C", "\u2510"          # ┌ ┐
BOX_SEP_L, BOX_SEP_R = "\u251C", "\u2524"        # ├ ┤
BOX_BOT_L, BOX_BOT_R = "\u2514", "\u2518"        # └ ┘
BOX_TICK_L, BOX_TICK_R = "\u251C", "\u2524"      # ├ ┤ (axis-label band)
HLINE, VLINE = "\u2500", "\u2502"                # ─ │
TICK_MAJ, TICK_MIN = "\u2534", "\u252C"          # ┴ ┬


def bin_index(frac_of_span):
    """position within the tested span (0..1) -> bin 0..63, clamped."""
    return min(BINS - 1, max(0, int(frac_of_span * BINS)))


def misses_to_bins(misses, needles, embed_fracs):
    """misses: set of planted words NOT recovered; needles: [(w,num),...].
    embed_fracs: each needle's position within the haystack (0..1 of span)."""
    mb = [0] * BINS
    for (w, _n), f in zip(needles, embed_fracs):
        if w in misses:
            mb[bin_index(f)] += 1
    return mb


def _cell(m, line):
    ru = ROWS - PACK * line
    rl = ru - 1
    return BLOCK[(1 if m >= ru else 0) | (2 if m >= rl else 0)]


def _row(gutter, ch_l, body, ch_r):
    """THE single row constructor: GUT gutter + left char + body + right."""
    assert len(gutter) == GUT, f"gutter {gutter!r} not {GUT} wide"
    assert len(body) == BINS, f"body {body!r} not {BINS} wide"
    return gutter + ch_l + body + ch_r


def _tok_label(tok):
    if tok >= 1048576 and tok % 1048576 == 0:
        return f"{tok // 1048576}M"
    if tok >= 1024 and tok % 1024 == 0:
        return f"{tok // 1024}K"
    if tok >= 1024:
        return f"{tok / 1024:.0f}K"
    return str(tok)


MIN_STEP = 8 * 1024                        # tick increment floor: 8K


def _milestones(span):
    """Binary milestone ticks for a linear axis, e.g. span=768K ->
    [64K,128K,256K,512K] (the span end is added by the caller).

    Base = smallest 8K*2^k with base*12 >= span (keeps ~5-6 labels);
    never below 8K per the user's minimum-increment rule.
    """
    base = MIN_STEP
    while base * 12 < span:
        base *= 2
    ms, x = [], base
    while x < span:
        ms.append(x)
        x *= 2
    return ms


def _check(lines, width):
    for x in lines:
        assert len(x) == width, f"width {len(x)} != {width}: {x!r}"
        assert all(ord(ch) in ALLOWED for ch in x), \
            f"forbidden glyph in framed line: {x!r}"


def render(misses_by_bin, ctx_tokens, model="", seed=None, elapsed_s=None,
           depth=DEPTH):
    span = int(ctx_tokens * depth)
    bin_tok = span // BINS
    total = sum(misses_by_bin)
    width = GUT + 1 + BINS + 1
    blank = " " * GUT
    lines = []

    def hdr(t):
        return _row(blank, VLINE, t[:BINS].ljust(BINS), VLINE)

    lines.append(_row(blank, BOX_TOP, HLINE * BINS, BOX_TOPR))    # ┌─…─┐
    h1 = f" {model}  MISSED needles   span = 0-{int(depth*100)}% of ctx"
    h2 = (" ctx={:,}".format(ctx_tokens)
          + (f"  seed={seed}" if seed is not None else "")
          + (f"  {elapsed_s:.0f}s" if elapsed_s is not None else ""))
    h3 = (f" missed {total}   bin={bin_tok:,} tok   1 dot=1 needle"
         f"   ceiling={FULL}")
    for t in (h1, h2, h3):
        lines.append(hdr(t))
    lines.append(_row(blank, BOX_SEP_L, HLINE * BINS, BOX_SEP_R))  # ├─…─┤

    for L in range(ROWS // PACK):
        top = ROWS - PACK * L
        lab = f"{top:>{GUT}}" if top % 8 == 0 else blank
        bars = "".join(_cell(misses_by_bin[b], L) for b in range(BINS))
        lines.append(_row(lab, VLINE, bars, VLINE))                 # bars

    # ticks at token milestones (>=8K increments) mapped onto the linear
    # span, plus the forced span-end tick. This rule IS the box bottom:
    # corners close the border here, nothing is bordered below it.
    ms = _milestones(span)
    ms_cols = {round(t / span * BINS) for t in ms if t < span}
    last = BINS - 1
    tick_body = "".join(
        TICK_MAJ if (b in ms_cols or b == 0 or b == last) else
        (TICK_MIN if b % 4 == 0 else HLINE)
        for b in range(BINS))
    lines.append(_row(blank, BOX_BOT_L, tick_body, BOX_BOT_R))     # └─axis─┘
    _check(lines, width)

    # ---- token labels BELOW the axis, unbordered (aligned to the grid) --
    grid = [" "] * (BINS + GUT + 1)
    edge = BINS + GUT

    def stamp(col, s):
        start = min(max(GUT + 1, col - (len(s) - 1)), edge - len(s) + 1)
        for i, chx in enumerate(s):
            grid[start + i] = chx

    stamp(GUT + 1, "0")                                   # origin
    for t in ms:
        stamp(GUT + 1 + round(t / span * BINS), _tok_label(t))
    stamp(GUT + 1 + last, _tok_label(span))      # span end (768K @ 1M ctx)
    lines.append("".join(grid).rstrip())
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="missed-needle box histogram")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--ctx", type=int, default=1048576)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    rng = random.Random(a.seed)
    misses = []
    for b in range(BINS):
        rate = 0.02 + (0.55 * max(0, b - 52) / 12.0)     # tail-collapse demo
        misses.append(sum(1 for _ in range(FULL) if rng.random() < rate))
    print(render(misses, a.ctx, model="ds4f-gb10", seed=a.seed,
                 elapsed_s=2412 if a.demo else None))


if __name__ == "__main__":
    main()
