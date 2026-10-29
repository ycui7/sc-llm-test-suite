# Action log — table-alignment check (2026-10-29)

Requested: add an item to the LLM test suites that asks a model to generate
an ASCII table (multiple columns/rows, unequal column widths, non-repeating
variable-length cells) and reports how many lines are misaligned. No
pass/fail bar — the count is the result.

Followed ub-mode (Make it Work → Make it Right → Practice).

1. 08:16 Explored both suite dirs; `sc-llm-test-suite` is the live git-tracked
   one (remote github.com/ycui7). Added the item there, not the frozen
   `llm-test-suites` copy.
2. 08:16 Read qc_common.py, run_all.py, test_coherence.py, README.md to match
   house style (stdlib-only, make_arg_parser, --save-result, qc_* helpers).
3. 08:20 TDD red: wrote tests/test_qc_table.py first (9 behavior tests on the
   pure measurer, no network). Saw it fail for the right reason
   (ModuleNotFoundError: qc_table).
4. 08:22 Wrote qc_table.py — pure measurement, text in / numbers out
   (boundary discipline: the endpoint I/O stays in the test script).
   Design decision: the reference pattern is the MOST FREQUENT delimiter
   pattern (never "the header"), because models deviate anywhere; ties break
   by first appearance so runs are deterministic.
5. 08:24 Two honest test failures, both mine:
   - grid tables (`+---+` borders carry no '|') were excluded from extraction
     → extraction now keeps border lines; delimiter is '+' on a border line,
     '|' on a pipe line. Root cause fixed in the measurer, not the test.
   - my fixture claimed 1 misaligned row but had 2 → fixed the fixture.
6. 08:26 Third failure: rules line `|---|---|` deviates from the data-row
   majority too. Kept the measurer's behavior (misaligned_lines counts every
   deviating table line; misaligned_rows counts only data rows) and pinned
   it with the assertion. Both numbers are reported.
7. 08:27 9/9 unit tests OK.
8. 08:28 Wrote test_table_alignment.py: prompt with the four pinned rules
   (unequal widths, unique cells, variable length, bars lined up), measures
   via qc_table, prints REPORT + JSON, ALWAYS exit 0 (informational).
   Flags: --rows/--columns/--rounds/--seed (pinned seed → reproducible).
9. 08:30 Wired into run_all.py as step 7 with INFORMATIONAL set: verdict shows
   INFO and never gates VERIFIED. README table row 7 added.
10. 08:31 Live end-to-end probe through Bifrost (sc-hub:4000) on
    gx10/qwen38fn-fp4-qad, 2 rounds, seed 7 — result appended below.

## End-to-end result

(seen live in this session; appended after the probe exited)
