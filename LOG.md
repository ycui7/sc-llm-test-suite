# Action log — table-alignment check (2026-10-29)

Requested: add an item to the LLM test suites that asks a model to generate
an ASCII table (multiple columns/rows, unequal column widths, non-repeating
variable-length cells) and reports how many lines are misaligned. No
pass/fail bar — the count is the result.

Followed ub-mode (Make it Work → Make it Right → Practice).

1. 08:16 Explored both suite dirs; `sc-llm-test-suite` is the live git-tracked
   one (tracked on GitHub, personal account). Added the item there, not the frozen
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
10. 08:31 Live end-to-end probe through the fleet Bifrost gateway on
    gx10/qwen38fn-fp4-qad, 2 rounds, seed 7 — result appended below.

## End-to-end result

Live through the fleet Bifrost gateway, model gx10/qwen38fn-fp4-qad,
--rounds 2 --seed 7 (JSON: scratch/table-align-e2e.json):

- round 0: 6 cols, widths [31,7,15,48,10,14], 9 data rows, 0 duplicate
  cells, MISALIGNED LINES 1 (idx [0] — the header row).
- round 1: 6 cols, widths [20,7,13,27,8,26], 9 data rows, 0 duplicate
  cells, MISALIGNED LINES 0.
- REPORT: tables found 2/2 | misaligned lines 1 | misaligned rows 1.

run_all.py integration (endpoint + table-alignment, rest skipped):
verdict shows `table-alignment INFO`, model still VERIFIED, exit 0.
- 2026-11-06: added --reasoning-effort to test_needles.py. Repro: 2048-needle run on gx10/glm-5.3-exl3 stalled at 0/2048 (no SSE chunk >300s; engine thinks silently, ignores chat_template_kwargs). Fix evidence: re-run recall 2048/2048, 0 dup, single inference, 787s (results/glm53-exl3-1106_0910 + run proc_d76dbe882575).

# Action log — config hygiene + docs/code sync (2026-11-26)

Followed ub-mode (Make it Work -> Make it Right -> Practice).

1. Scripts now take the serving endpoint from QC_BASE_URL / --base-url and
   fail loudly when unset (qc_common.openai_base, run_all, battery
   ${QC_BASE_URL:?}). Credentials are read from a chmod-600 file
   (~/.config/qc/apikey). llama-benchy is discovered via QC_BENCHY /
   ~/.venv/bin / PATH; scratch default via expanduser.
2. README gate table aligned with run_all STEPS (8 steps incl. prefix-cache
   step 5); standalone gates/helpers documented (qc_4b_tool_file real-file
   round trip, test_ssd_restore, grade_needles_offline, qc_coverage,
   qc_hist, qc_stream).
3. Needle spec text aligned with the 2026-10-05 directive: span =
   min(2/3 ctx, 512K), rule-4 output budget (supersedes the 3/4-window
   wording in README / QUALIFICATION_2H_PLAN / REQUIREMENTS_histogram).
4. test_speed: concurrency-scaling gate — c2 aggregate tg must exceed
   1.5x c1, else the bench aborts early and the gate FAILS.
5. test_coherence: refusal matcher accepts the "not able to" family
   (verified refusals were false-failing).
6. New helper qc_coverage.py: per-depth-bin placed-vs-found chart from a
   real run's artifacts.
7. Verified: unit tests 9/9; py_compile clean.
