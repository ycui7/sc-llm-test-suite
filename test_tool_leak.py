#!/usr/bin/env python3
"""Check 4: no tool-calling leak.

With tools present in the schema, a plain non-tool question must come back as
normal prose — NOT as raw tool-call JSON in content, not a prompt echo, not a
fence-dump of the schema. Also verifies a REAL tool round-trip works: the
model should emit a well-formed tool_call for a question that needs it.

Usage: python3 test_tool_leak.py [--base-url URL] [--model ID]
"""
import json
import re

from qc_common import chat, last_answer, make_arg_parser, resolve_model, Result

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "offset_c": {"type": "number", "description": "Celsius offset"},
            },
            "required": ["city"],
        },
    },
}]

LEAK_JSON = re.compile(
    r'\{\s*"?(tool_call|name|function|arguments)"?\s*[:,\}]|<\|tool_call|<tool_call>',
    re.I)


def main():
    args = make_arg_parser("tool-call leak check").parse_args()
    r = Result("tool-leak")
    args.model = resolve_model(args.base_url, args.api_key, args.model)

    # 1) plain question WITH tools in schema -> prose answer, no tool JSON
    try:
        out = chat(args.base_url, args.api_key, args.model,
                   [{"role": "user",
                     "content": "What is 12 * 12? Answer with just the number."}],
                   max_tokens=args.max_tokens, timeout=args.timeout, tools=TOOLS)
        ans = last_answer(out["content"], out["reasoning"])
        leaked = bool(LEAK_JSON.search(out["content"]))
        echo = "What is 12 * 12" in ans and "144" not in ans
        r.add("plain-with-tools", (not leaked) and (not echo) and "144" in ans,
              f"finish={out['finish']} content={out['content'][:100]!r}")
    except Exception as e:  # noqa: BLE001
        r.add("plain-with-tools", False, repr(e)[:400])

    # 2) question that REQUIRES the tool -> well-formed tool_call emitted
    try:
        out = chat(args.base_url, args.api_key, args.model,
                   [{"role": "user", "content":
                     "What is the weather in Paris? Use the get_weather tool."}],
                   max_tokens=args.max_tokens, timeout=args.timeout, tools=TOOLS)
        calls = out["tool_calls"]
        ok = False
        detail = "no tool_calls"
        if calls:
            fn = calls[0].get("function", {})
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                a = {}
            ok = (fn.get("name") == "get_weather"
                  and "paris" in str(a.get("city", "")).lower())
            detail = f"name={fn.get('name')} args={fn.get('arguments')!r}"
        r.add("tool-emitted", ok, detail)
    except Exception as e:  # noqa: BLE001
        r.add("tool-emitted", False, repr(e)[:400])

    # 3) answer AFTER a real tool result -> uses the returned value
    try:
        conv = [{"role": "user", "content": "What is the weather in Paris?"}]
        out = chat(args.base_url, args.api_key, args.model, conv,
                   max_tokens=args.max_tokens, timeout=args.timeout, tools=TOOLS)
        if out["tool_calls"]:
            tc = out["tool_calls"][0]
            conv.append({"role": "assistant", "content": out["content"] or None,
                         "tool_calls": [tc]})
            conv.append({"role": "tool", "tool_call_id": tc.get("id", "call_0"),
                         "content": json.dumps({"temp_c": 20})})
            out2 = chat(args.base_url, args.api_key, args.model, conv,
                        max_tokens=args.max_tokens, timeout=args.timeout,
                        tools=TOOLS)
            ans = last_answer(out2["content"], out2["reasoning"])
            r.add("tool-result-consumed", "20" in ans, f"answer={ans[:120]!r}")
        else:
            r.add("tool-result-consumed", False, "model never called the tool")
    except Exception as e:  # noqa: BLE001
        r.add("tool-result-consumed", False, repr(e)[:400])

    r.finish(args)


if __name__ == "__main__":
    main()
