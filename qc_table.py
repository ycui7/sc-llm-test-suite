"""Pure measurement of ASCII-table column alignment in model output.

Used by test_table_alignment.py; unit-tested in tests/test_qc_table.py.
No I/O here: text in, measurements out.

Definitions (glossary):
  table line      -- an extracted line: either a pipe line (carries a vertical
                     bar) or a border line (carries '+' and only border chars).
  delimiter       -- the character that marks a column edge: '|' on a pipe
                     line, '+' on a border line.
  bar pattern     -- the tuple of character offsets of the delimiters in a
                     line, after tabs are expanded (8 cols) and box-drawing
                     verticals are folded to '|'.
  reference       -- the most frequent bar pattern among the table lines
                     (majority wins, never the first line).
  misaligned      -- a table line whose bar pattern differs from the reference.
  separator       -- a pipe line whose cells hold only dashes/equals/colons.
  data row        -- a table line that is neither a separator nor a border
                     line (the header row counts as one).
"""
from collections import Counter

TAB_STOP = 8
_VERTS = {"\u2502": "|", "\u2551": "|", "\uff5c": "|"}  # │ ║ ｜
_SEP_CHARS = set("-=:_\u2500\u2501\u254c\u2550 \t")
_BORDER_CHARS = set("+-=:\u2500\u2501\u254c\u2550\t ")
_DASHES = "-=\u2500\u2501\u2550"


def _has_bar(line):
    return "|" in line or any(v in line for v in _VERTS)


def _is_border_line(line):
    return "+" in line and set(line) <= _BORDER_CHARS


def extract_table_lines(text):
    """Return the table-looking lines (pipe lines and grid border lines).

    Trailing blanks are dropped (they never affect alignment); everything
    else is preserved verbatim so measurement sees the raw layout.
    """
    out = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if _has_bar(line) or _is_border_line(line):
            out.append(line)
    return out


def _normalize(line):
    for src, dst in _VERTS.items():
        line = line.replace(src, dst)
    return line.expandtabs(TAB_STOP)


def _delimiter_positions(line):
    ch = "+" if _is_border_line(line) else "|"
    return tuple(i for i, c in enumerate(line) if c == ch)


def _cells(line):
    parts = [c.strip() for c in line.split("|")]
    if parts and parts[0] == "":
        parts.pop(0)
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _is_separator(line):
    """A pipe line whose cells hold only filler (a rules row), not data."""
    if not _has_bar(line):
        return False
    cells = _cells(line)
    if not cells:
        return False
    has_dash = any(d in line for d in _DASHES)
    return has_dash and all(set(c) <= _SEP_CHARS for c in cells)


def _column_widths(ref_pattern):
    return [ref_pattern[i + 1] - ref_pattern[i]
            for i in range(len(ref_pattern) - 1)]


def _duplicate_cell_count(lines, data_idx):
    seen = Counter()
    for i in data_idx:
        for cell in _cells(lines[i]):
            if cell:
                seen[cell] += 1
    return sum(1 for _, n in seen.items() if n > 1)


def measure_alignment(text):
    """Measure bar alignment in `text`. See module docstring for terms.

    Returns a dict of measurements only — no verdict. found=False (with
    None measurements) when fewer than two bar lines are present.
    """
    lines = [_normalize(l) for l in extract_table_lines(text)]
    empty = {
        "found": False, "bar_lines": 0, "rows": 0, "columns": 0,
        "misaligned_lines": None, "misaligned_rows": None,
        "misaligned_line_indices": None, "reference_pattern": None,
        "column_widths": None, "uniform_widths": None,
        "duplicate_cells": None,
    }
    if len(lines) < 2:
        return empty

    patterns = [_delimiter_positions(l) for l in lines]
    counts = Counter(patterns)
    first_seen = {}
    for i, p in enumerate(patterns):
        first_seen.setdefault(p, i)
    # Majority wins; ties break by first appearance (deterministic).
    reference = min(counts, key=lambda p: (-counts[p], first_seen[p]))
    mis_idx = [i for i, p in enumerate(patterns) if p != reference]
    data_idx = [i for i, l in enumerate(lines)
                if _has_bar(l) and not _is_separator(l)]
    mis_rows = [i for i in data_idx if patterns[i] != reference]
    widths = _column_widths(reference)

    return {
        "found": True,
        "bar_lines": len(lines),
        "rows": len(data_idx),
        "columns": max(len(reference) - 1, 0),
        "misaligned_lines": len(mis_idx),
        "misaligned_rows": len(mis_rows),
        "misaligned_line_indices": mis_idx,
        "reference_pattern": list(reference),
        "column_widths": widths,
        "uniform_widths": (len(set(widths)) == 1) if len(widths) >= 2 else None,
        "duplicate_cells": _duplicate_cell_count(lines, data_idx),
    }
