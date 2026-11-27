#!/usr/bin/env python3
"""Grade a saved needle answer three ways, so a FAIL can be attributed.

The harness bar is strict and all-or-nothing (ok = len(hits) == n) and its
matcher accepts only "word: 12345". That conflates three very different
outcomes, so report each separately:

  strict    the harness's own matcher  -- the official number
  tolerant  correct number in any formatting (markdown, wrap, spacing)
  presence  the word appears at all    -- retrieved vs never surfaced
"""
import collections
import json
import re
import sys

answer_path, needles_path = sys.argv[1], sys.argv[2]
ans = open(answer_path, encoding="utf-8", errors="ignore").read()
needles = [tuple(x) for x in json.load(open(needles_path))]
expected = {w: int(k) for w, k in needles}
n = len(needles)

# exactly the harness's LINE_RE
STRICT = re.compile(r"(?<![-\w])([A-Za-z]+(?:-[A-Za-z]+)*):\s*(\d{5})")
strict_hits = {m.group(1) for m in STRICT.finditer(ans)
               if expected.get(m.group(1)) == int(m.group(2))}

# heal the formatting the strict matcher cannot see: hyphen line-wraps, markdown
# emphasis around the token, spaced or comma-grouped digits
flat = ans.replace("-\n", "-")
tol_hits, wrong_number, absent = set(), [], []
for w, k in needles:
    if w in strict_hits:
        tol_hits.add(w)
        continue
    pat = re.compile(r"(?<![\w-])[*_`\s]*(" + re.escape(w) + r")[*_`\s]*[:=\-–]\s*[*_`\s]*(\d[\d ,]{3,10})",
                     re.IGNORECASE)
    got = None
    for m in pat.finditer(flat):
        digits = re.sub(r"[^0-9]", "", m.group(2))[:5]
        if len(digits) == 5:
            got = int(digits)
            break
    if got is None:
        absent.append(w)
    elif got == k:
        tol_hits.add(w)
    else:
        wrong_number.append((w, k, got))

present = {w for w in expected
           if re.search(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", ans, re.IGNORECASE)}
never = [w for w in expected if w not in present]

print(f"needles {n}   answer {len(ans)} chars")
print(f"1) STRICT   harness matcher                 {len(strict_hits)}/{n}  {len(strict_hits)/n*100:.2f}%")
print(f"2) TOLERANT any format, right number        {len(tol_hits)}/{n}  {len(tol_hits)/n*100:.2f}%")
print(f"3) PRESENCE word appears at all             {len(present)}/{n}  {len(present)/n*100:.2f}%")
print(f"\n  formatting-only recoveries : {len(tol_hits) - len(strict_hits)}")
print(f"  number written wrong       : {len(wrong_number)}  {wrong_number[:6]}")
print(f"  never surfaced             : {len(never)}  {never[:12]}")
suf = collections.Counter(w.split("-")[-1] for w, _ in needles if w not in strict_hits)
print(f"  strict misses by suffix    : {dict(suf)}")
