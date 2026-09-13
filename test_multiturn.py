#!/usr/bin/env python3
"""Check 4b: multi-turn tool-call continuity (agentic loop).

Drives the assistant/tool loop itself against fake tool backends and checks:
  C1 single tool + offset composition  (Paris +5 -> 20+5=25)
  C2 three sequential user turns in ONE conversation (context persistence)
  C3 two-tool compose + arithmetic on returned values (NY - London)

Backend tables are fixed so expected values are deterministic. NY baseline
is -2C (not -5) — verify checker values against the backend table before
trusting a FAIL.

Usage: python3 test_multiturn.py [--base-url URL] [--model ID]
"""
import json
import re

from qc_common import chat, last_answer, make_arg_parser, resolve_model, Result

WEATHER = {  # baseline temps in C
    "paris": 20, "tokyo": 15, "new york": -2, "london": 8,
}
TIME = {  # local hour offsets from UTC
    "tokyo": 9, "new york": -7, "london": 0, "paris": 2,
}
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current temp C for a city, plus offset.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "offset_c": {"type": "number", "description": "degrees C offset to add"},
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Current local hour (0-23) for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}},
]


def backend(name, a):
    city = str(a.get("city", "")).lower().strip()
    for k in WEATHER:
        if k in city:
            city = k
            break
    if name == "get_weather":
        base = WEATHER.get(city, 10)
        return {"city": city, "temp_c": base + float(a.get("offset_c", 0))}
    if name == "get_time":
        return {"city": city, "hour": (14 + TIME.get(city, 0)) % 24}
    return {"error": "unknown tool"}


def run_loop(args, conv, max_hops=8):
    """Drive tool-call hops until the model answers; returns (answer, n_tools, n_lm)."""
    n_tools = n_lm = 0
    for _ in range(max_hops):
        out = chat(args.base_url, args.api_key, args.model, conv,
                   max_tokens=args.max_tokens, timeout=args.timeout, tools=TOOLS)
        n_lm += 1
        if not out["tool_calls"]:
            return last_answer(out["content"], out["reasoning"]), n_tools, n_lm
        conv.append({"role": "assistant",
                     "content": out["content"] or "",
                     "tool_calls": out["tool_calls"]})
        for tc in out["tool_calls"]:
            fn = tc.get("function", {})
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                a = {}
            res = backend(fn.get("name", ""), a)
            n_tools += 1
            conv.append({"role": "tool", "tool_call_id": tc.get("id", "call"),
                         "content": json.dumps(res)})
    return "<no final answer>", n_tools, n_lm


def main():
    args = make_arg_parser("multi-turn tool continuity").parse_args()
    r = Result("multiturn")
    args.model = resolve_model(args.base_url, args.api_key, args.model)

    # C1: single tool + offset -> 25
    try:
        ans, nt, nl = run_loop(args, [{"role": "user", "content":
            "What is the temperature in Paris with a +5 C offset? "
            "Use get_weather and give the final number."}])
        n = re.findall(r"-?\d+(?:\.\d+)?", ans)
        r.add("C1-single-tool-offset", bool(n) and float(n[-1]) == 25,
              f"tools={nt} lm={nl} ans={ans[:100]!r}")
    except Exception as e:  # noqa: BLE001
        r.add("C1-single-tool-offset", False, repr(e)[:300])

    # C2: 3 user turns in ONE conversation (NY offset -3 -> -5; London hour 14)
    try:
        conv = []
        ok = True
        detail = []
        grades = []
        for q in ["Weather in Tokyo, then what time is it in Tokyo? Use the tools.",
                  "Now the weather in New York with a -3 C offset.",
                  "Finally, what hour of the day is it in London? Use get_time."]:
            conv.append({"role": "user", "content": q})
            ans, nt, nl = run_loop(args, conv)
            conv.append({"role": "assistant", "content": ans})
            ql = q.lower()
            if "tokyo" in ql:
                good = "15" in ans and "23" in ans
                why = f"want 15&23, got={ans[-200:]!r}" if not good else ""
            elif "new york" in ql:
                good = re.search(r"(-5)(?!\d)", ans) is not None
                why = f"want -5, got={ans[-200:]!r}" if not good else ""
            else:
                good = re.search(r"\b14\b", ans) is not None
                why = f"want 14, got={ans[-200:]!r}" if not good else ""
            ok = ok and good
            grades.append(f"{'ok' if good else 'BAD:' + why[:130]}")
            detail.append(f"[turn:{q[:12]}...]={ans[:120]!r}")
        # grade each turn on the FULL answer (details truncated only for display)
        r.add("C2-context-persistence", ok,
              "grades=" + ",".join(grades) + " | " + " ".join(detail)[:360])
    except Exception as e:  # noqa: BLE001
        r.add("C2-context-persistence", False, repr(e)[:300])

    # C3: two tools + compose -> -2 - 8 = -10
    try:
        ans, nt, nl = run_loop(args, [{"role": "user", "content":
            "What is the weather in New York minus the weather in London "
            "(temperature difference in C)? Use get_weather for each city and "
            "give the difference as a final number."}])
        n = re.findall(r"-?\d+(?:\.\d+)?", ans)
        r.add("C3-two-tool-compose", bool(n) and float(n[-1]) == -10,
              f"tools={nt} lm={nl} ans={ans[:100]!r}")
    except Exception as e:  # noqa: BLE001
        r.add("C3-two-tool-compose", False, repr(e)[:300])

    r.finish(args)


if __name__ == "__main__":
    main()
