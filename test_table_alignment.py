#!/usr/bin/env python3
"""Check 7: table-alignment probe (INFORMATIONAL — no pass/fail bar).

Asks the model to generate one ASCII table with several columns of unequal
width and several rows, cells filled with non-repeating content of variable
length. Measures how many table lines fail to line up with the majority
(bar-position) pattern and reports the count. The count is a datum, not a
verdict: this check never fails the deployment.

Measurement lives in qc_table.measure_alignment (unit-tested). This script
only drives the endpoint and reports.

Usage: python3 test_table_alignment.py [--base-url URL] [--model ID]
       [--rows 8] [--columns 6] [--rounds 1]
"""
import random

from qc_common import chat, last_answer, make_arg_parser, resolve_model
from qc_table import measure_alignment

PROMPT_TEMPLATE = """\
Build ONE plain-text ASCII table. Follow every rule exactly.

Rules:
1. {columns} columns and {rows} data rows, plus one header row.
2. The columns must NOT all be the same width. Give each column a different width.
3. Every cell must contain unique text. No text may repeat anywhere in the table.
4. Cell texts must vary a lot in length: some are one word, some are a long phrase.
5. Put a single space of padding on both sides of each cell.
6. Line every vertical bar | up with the bars in the other rows.
7. Use only ASCII characters. No Markdown pipes-only rows, no explanations.

Column names: {headers}
Output only the table.
"""


def build_prompt(rows, columns, seed):
    rng = random.Random(seed)
    headers = ["item", "qty", "origin", "note", "size", "status", "cost",
               "supplier"][:columns]
    while len(headers) < columns:
        headers.append(f"field{len(headers) + 1}")
    # Vary the wording slightly per round so rounds are not copies.
    p = PROMPT_TEMPLATE.format(columns=columns, rows=rows,
                               headers=", ".join(headers))
    if rng.random() < 0.5:
        p = p.replace("Output only the table.",
                      "Return nothing but the table itself.")
    return p


def main():
    p = make_arg_parser("table-alignment probe (informational)")
    p.add_argument("--rows", type=int, default=8,
                   help="data rows requested (default: %(default)s)")
    p.add_argument("--columns", type=int, default=6,
                   help="columns requested (default: %(default)s)")
    p.add_argument("--rounds", type=int, default=1,
                   help="independent generations (default: %(default)s)")
    p.add_argument("--seed", type=int, default=None,
                   help="pinned prompt seed for reproducibility")
    args = p.parse_args()
    args.model = resolve_model(args.base_url, args.api_key, args.model)
    rng = random.Random(args.seed if getattr(args, "seed", None) is not None
                        else 20261029)

    rounds = []
    for i in range(args.rounds):
        prompt = build_prompt(args.rows, args.columns, rng.randrange(1 << 30))
        try:
            out = chat(args.base_url, args.api_key, args.model,
                       [{"role": "user", "content": prompt}],
                       max_tokens=args.max_tokens, timeout=args.timeout)
            ans = last_answer(out["content"], out["reasoning"])
            m = measure_alignment(ans)
            rounds.append({"round": i, "finish": out["finish"],
                           "elapsed_s": round(out["elapsed_s"], 1),
                           "prompt_chars": len(prompt),
                           "answer_chars": len(ans), **m})
            print(f"--- round {i} ---")
            print(f"  table found      : {m['found']}")
            if m["found"]:
                print(f"  table lines      : {m['bar_lines']}")
                print(f"  data rows        : {m['rows']}")
                print(f"  columns          : {m['columns']}")
                print(f"  column widths    : {m['column_widths']}"
                      f"{' (UNIFORM — prompt asked for unequal)' if m['uniform_widths'] else ''}")
                print(f"  duplicate cells  : {m['duplicate_cells']}")
                print(f"  MISALIGNED LINES : {m['misaligned_lines']}")
                print(f"  MISALIGNED ROWS  : {m['misaligned_rows']}")
                print(f"  misaligned idx   : {m['misaligned_line_indices']}")
            else:
                print("  MISALIGNED LINES : n/a (no table in the answer)")
                print("  answer head      : " + repr(ans[:200]))
        except Exception as e:  # noqa: BLE001
            rounds.append({"round": i, "error": str(e)[:300], "found": False})
            print(f"--- round {i} ---\n  ERROR: {e}")

    measured = [r for r in rounds if r.get("found")]
    summary = {
        "suite": "table-alignment",
        "informational": True,
        "note": "no pass/fail bar — the misaligned count is the result",
        "model": args.model,
        "base_url": args.base_url,
        "requested_rows": args.rows,
        "requested_columns": args.columns,
        "rounds": rounds,
        "misaligned_lines_total": sum(r.get("misaligned_lines") or 0
                                      for r in measured),
        "misaligned_rows_total": sum(r.get("misaligned_rows") or 0
                                     for r in measured),
        "tables_found": len(measured),
        "rounds_run": len(rounds),
    }
    print(f"\nREPORT {args.model}: tables found {len(measured)}/{args.rounds}"
          f" | misaligned lines {summary['misaligned_lines_total']}"
          f" | misaligned rows {summary['misaligned_rows_total']}")
    if args.save_result:
        import json
        with open(args.save_result, "w") as f:
            json.dump(summary, f, indent=2)
        print("saved: " + args.save_result)
    raise SystemExit(0)  # informational: never fails the run


if __name__ == "__main__":
    main()
