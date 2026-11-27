#!/usr/bin/env python3
"""Check 4b: multi-turn tool-call continuity (agentic loop).

Drives the assistant/tool loop itself against fake tool backends and checks:
  C1 single tool + offset composition  (Paris +5 -> 20+5=25)
  C2 three sequential user turns in ONE conversation (context persistence)
  C3 two-tool compose + arithmetic on returned values (NY - London)
  C4 three tool-calling ROUNDS in one conversation with preserve_thinking
     enabled and the thinking block echoed back on rounds 2 and 3

C4 is the agentic-harness contract: a thinking model keeps its reasoning in
history across tool rounds. It also guards the failure mode where the engine
leaks a text-form tool block into `content` and returns no structured
tool_calls (harness then reports an incomplete/malformed tool call).

Backend tables are fixed so expected values are deterministic. NY baseline
is -2C (not -5) — verify checker values against the backend table before
trusting a FAIL.

Usage: python3 test_multiturn.py [--base-url URL] [--model ID]
       [--no-preserve-thinking] [--allow-no-reasoning]
"""
import json
import re

def _num(text):
    """Last number in text, normalizing unicode minus/plus so grader matches."""
    t = text.replace("\u2212", "-").replace("\u2213", "+").replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return float(nums[-1]) if nums else None

def _nums(text):
    """All numbers in text (unicode-normalized) — graders must not assume the
    final number is the answer: chatty answers ("**25.0 °C** ... +5 °C offset")
    end on an echo of an input value."""
    t = text.replace("\u2212", "-").replace("\u2213", "+").replace(",", "")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]

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

# Text-form tool markup: if this shows up in `content` while tool_calls is
# empty, the engine failed to parse its own tool block (harness sees an
# unterminated/incomplete tool call).
TEXT_TOOL_MARKERS = ("<tool_call>", "<function=", "<|tool_calls|>")


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


def _bad_args(tool_calls):
    """Tool calls whose arguments are not valid JSON (the classic harness break)."""
    bad = []
    for tc in tool_calls:
        fn = tc.get("function", {}) or {}
        try:
            json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            bad.append(fn.get("name", "?") + ":" + str(fn.get("arguments"))[:60])
    return bad


def _text_tool_leak(content):
    return [m for m in TEXT_TOOL_MARKERS if m in (content or "")]


def run_loop(args, conv, max_hops=8, preserve=False, keep_reasoning=None):
    """Drive tool-call hops until the model answers.

    preserve=True  -> requests chat_template_kwargs {enable_thinking,
                      preserve_thinking} and echoes the assistant thinking block
                      back as `reasoning_content` on the next request.
    keep_reasoning -> list; when given, every assistant reasoning block is
                      appended to it (proof of what was sent back).

    Returns (answer, n_tools, n_lm, reasonings, leaks, bad_args).
    """
    n_tools = n_lm = 0
    reasonings, leaks, badargs = [], [], []
    extra = args.think_kwargs if preserve else None
    for _ in range(max_hops):
        out = chat(args.base_url, args.api_key, args.model, conv,
                   max_tokens=args.max_tokens, timeout=args.timeout,
                   tools=TOOLS, extra=extra)
        n_lm += 1
        if out["reasoning"]:
            reasonings.append(out["reasoning"])
            if keep_reasoning is not None:
                keep_reasoning.append(out["reasoning"])
        if not out["tool_calls"]:
            leaked = _text_tool_leak(out["content"])
            if leaked:
                leaks.append(leaked)
            return (last_answer(out["content"], out["reasoning"]),
                    n_tools, n_lm, reasonings, leaks, badargs)
        badargs += _bad_args(out["tool_calls"])
        leaked = _text_tool_leak(out["content"])
        if leaked:
            leaks.append(leaked)
        msg = {"role": "assistant",
               "content": out["content"] or "",
               "tool_calls": out["tool_calls"]}
        if preserve and out["reasoning"]:
            # The thinking block goes back on the wire, like a harness replays it.
            msg["reasoning_content"] = out["reasoning"]
        conv.append(msg)
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
    return "<no final answer>", n_tools, n_lm, reasonings, leaks, badargs


def main():
    p = make_arg_parser("multi-turn tool continuity")
    p.add_argument("--no-preserve-thinking", action="store_true",
                   help="run C4 without preserve_thinking / without replaying the "
                        "thinking block (control run)")
    p.add_argument("--allow-no-reasoning", action="store_true",
                   help="do not fail C4 when the endpoint returns no reasoning "
                        "(non-thinking model)")
    p.add_argument("--think-kwargs", default=None,
                   help='JSON override for chat_template_kwargs in C4 '
                        '(default: {"enable_thinking": true, "preserve_thinking": true})')
    args = p.parse_args()
    args.think_kwargs = json.loads(args.think_kwargs) if args.think_kwargs else {
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    r = Result("multiturn")
    args.model = resolve_model(args.base_url, args.api_key, args.model)

    # C1: single tool + offset -> 25
    try:
        ans, nt, nl, _rs, _lk, _ba = run_loop(args, [{"role": "user", "content":
            "What is the temperature in Paris with a +5 C offset? "
            "Use get_weather and give the final number."}])
        ok = any(abs(x - 25.0) < 1e-6 for x in _nums(ans))
        r.add("C1-single-tool-offset", ok,
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
            ans, nt, nl, _rs, _lk, _ba = run_loop(args, conv)
            conv.append({"role": "assistant", "content": ans})
            ql = q.lower()
            if "tokyo" in ql:
                good = "15" in ans and "23" in ans
                why = f"want 15&23, got={ans[-200:]!r}" if not good else ""
            elif "new york" in ql:
                good = re.search(r"(?:\u2212|-)5(?!\d)", ans) is not None
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

    # C3: two tools + compose -> -2 - 8 = -10. Grade ANY of the composed values
    # (-10 / -10.0 / "10 degrees colder") via signed scan: find the number that
    # immediately follows a minus/difference context, else accept -10 anywhere.
    # The old grader took the LAST number in the text, which a chatty answer
    # breaks ("... -10.0C (New York is 10C colder than London)" -> 10.0).
    try:
        ans, nt, nl, _rs, _lk, _ba = run_loop(args, [{"role": "user", "content":
            "What is the weather in New York minus the weather in London "
            "(temperature difference in C)? Use get_weather for each city and "
            "give the difference as a final number."}])
        t = ans.replace("\u2212", "-").replace("\u2213", "+")
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
        # the composed value: a -10.0 token, or -10 signed anywhere, or the
        # pattern "<a> - <b> = -10" as a plain arithmetic line
        ok_c3 = any(abs(x + 10.0) < 1e-6 for x in nums) or \
                re.search(r"=\s*-10(?:\.0+)?\b", t) is not None
        r.add("C3-two-tool-compose", ok_c3 and nt >= 2,
              f"tools={nt} lm={nl} nums={nums[:8]} ans={ans[:100]!r}")
    except Exception as e:  # noqa: BLE001
        r.add("C3-two-tool-compose", False, repr(e)[:300])

    # C4: three tool-calling ROUNDS, one conversation, thinking preserved.
    # r1 Paris+5 -> 25 · r2 +Tokyo 15 -> 40 · r3 -London 8 -> 32
    try:
        preserve = not args.no_preserve_thinking
        conv = []
        expected = [25.0, 40.0, 32.0]
        rounds = [
            "Get the weather in Paris with a +5 C offset. Use get_weather.",
            "Add that Paris result to the temperature in Tokyo. Use get_weather for Tokyo.",
            "Now subtract the temperature in London from your running total. "
            "Use get_weather for London and give the final number.",
        ]
        sent_thinking = []          # reasoning blocks seen so far
        hist_counts = []            # assistant msgs carrying reasoning at round start
        per_round = []
        ok_all = True
        for idx, q in enumerate(rounds, 1):
            conv.append({"role": "user", "content": q})
            hist_counts.append(sum(1 for m in conv if m.get("reasoning_content")))
            ans, nt, nl, rs, leaks, badargs = run_loop(
                args, conv, preserve=preserve, keep_reasoning=sent_thinking)
            good_num = any(abs(x - expected[idx - 1]) < 1e-6 for x in _nums(ans))
            good_call = nt >= 1
            good_json = not badargs
            good_no_leak = not leaks
            ok = good_num and good_call and good_json and good_no_leak
            ok_all = ok_all and ok
            per_round.append(
                f"r{idx}:{'ok' if ok else 'BAD'}"
                f"(num={good_num},call={good_call},json={good_json},"
                f"leak={good_no_leak},tools={nt},lm={nl},"
                f"think_recv={sum(len(x) for x in rs)},"
                f"hist_think={hist_counts[-1]})")
            # The assistant's visible answer joins the history; keep its thinking
            # block too so the next round sees the full assistant turn.
            final = {"role": "assistant", "content": ans}
            if preserve and rs:
                final["reasoning_content"] = rs[-1]
            conv.append(final)
        r.add("C4-three-round-preserve-thinking", ok_all,
              ("preserve=%s " % preserve) + " ".join(per_round)[:520])

        # C4b: the thinking block really is on the wire for rounds 2 and 3 —
        # round 2 must see round 1's, round 3 must see rounds 1+2.
        if preserve:
            ok_echo = (len(hist_counts) == 3
                       and hist_counts[1] >= 1 and hist_counts[2] >= 2)
            r.add("C4b-thinking-echoed-r2-r3", ok_echo,
                  f"assistant msgs with reasoning_content at round start="
                  f"{hist_counts} (need r2>=1, r3>=2)")
            if not sent_thinking and not args.allow_no_reasoning:
                r.add("C4c-thinking-returned", False,
                      "endpoint returned no reasoning_content with "
                      "enable_thinking+preserve_thinking — preserve_thinking is "
                      "untestable on this model (pass --allow-no-reasoning to waive)")
            elif not sent_thinking:
                r.add("C4c-thinking-returned", True,
                      "waived via --allow-no-reasoning (no reasoning_content)")
            else:
                r.add("C4c-thinking-returned", True,
                      f"reasoning returned in {len(sent_thinking)} hop(s), "
                      f"len={[len(x) for x in sent_thinking]}")
    except Exception as e:  # noqa: BLE001
        r.add("C4-three-round-preserve-thinking", False, repr(e)[:300])
        r.add("C4b-thinking-echoed-r2-r3", False, repr(e)[:300])

    r.finish(args)


if __name__ == "__main__":
    main()
