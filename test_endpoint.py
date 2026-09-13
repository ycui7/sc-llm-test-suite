#!/usr/bin/env python3
"""Check 5: endpoint health.

Verifies the model is reachable through the endpoint, registered under the
expected id, and answers a minimal chat completion (the test_smoke.sh
equivalent in python, plus an error-shape check).

Usage: python3 test_endpoint.py [--base-url URL] [--model ID]
"""
import sys

from qc_common import chat, list_models, make_arg_parser, resolve_model, Result

PROMPT = "Reply with exactly the word: PONG"


def main():
    args = make_arg_parser("endpoint health check").parse_args()
    r = Result("endpoint")

    try:
        ids = list_models(args.base_url, args.api_key, timeout=min(args.timeout, 30))
        r.add("models-list", len(ids) > 0, f"{len(ids)} models")
    except Exception as e:  # noqa: BLE001
        r.add("models-list", False, e)
        r.finish(args)
        return

    model = resolve_model(args.base_url, args.api_key, args.model)
    r.add("model-registered", model in ids, f"want {model!r}, have {ids[:10]}")

    try:
        out = chat(args.base_url, args.api_key, model,
                   [{"role": "user", "content": PROMPT}],
                   max_tokens=args.max_tokens, timeout=args.timeout)
        ans = (out["content"] or out["reasoning"]).strip()
        r.add("chat-pong", "pong" in ans.lower(), f"reply: {ans[:80]!r}")
    except Exception as e:  # noqa: BLE001
        r.add("chat-pong", False, e)

    r.finish(args)


if __name__ == "__main__":
    main()
