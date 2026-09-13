"""Shared helpers for the sc-llm-test-suite qualification scripts.

Zero third-party dependencies (stdlib only) so the suite runs on any box
with python3. Talks to any OpenAI-compatible endpoint (LiteLLM, vLLM,
sglang, llama-server).
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

# Defaults follow the standing conventions:
#  - all AI traffic goes through LiteLLM (localhost:4000)
#  - probes ALWAYS use a large max_tokens: thinking models empty their
#    content field when the budget is eaten by reasoning (4K hard floor,
#    32768 default).
DEFAULT_BASE_URL = os.environ.get("QC_BASE_URL", "http://localhost:4000/v1")
DEFAULT_API_KEY = os.environ.get("QC_API_KEY", os.environ.get("OPENAI_API_KEY"))
DEFAULT_MAX_TOKENS = int(os.environ.get("QC_MAX_TOKENS", "32768"))


def make_arg_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible base URL (default: %(default)s)")
    p.add_argument("--model", default=None,
                   help="model id; default: first model from /models")
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="output budget incl. reasoning (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-request timeout seconds")
    p.add_argument("--save-result", default=None,
                   help="write JSON result summary to this path")
    return p


# ---------------------------------------------------------------- requests
def http_json(base_url, path, api_key=None, payload=None, timeout=30.0):
    """GET (payload None) or POST json. Returns parsed dict; raises on HTTP error."""
    url = base_url.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def list_models(base_url, api_key, timeout=30.0):
    return [m["id"] for m in http_json(base_url, "/models", api_key, timeout=timeout)["data"]]


def resolve_model(base_url, api_key, model, timeout=30.0):
    if model:
        return model
    ids = list_models(base_url, api_key, timeout)
    if not ids:
        raise RuntimeError("endpoint returned no models")
    return ids[0]


def chat(base_url, api_key, model, messages, max_tokens=DEFAULT_MAX_TOKENS,
         timeout=600.0, temperature=0.0, tools=None, extra=None):
    """One non-streaming chat completion.

    Returns dict: content, reasoning, finish, usage, elapsed_s, ttft_s
    (ttft approximated by round-trip time for non-streaming calls).
    """
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if tools is not None:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    t0 = time.monotonic()
    resp = http_json(base_url, "/chat/completions", api_key, payload, timeout)
    elapsed = time.monotonic() - t0
    try:
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish": choice.get("finish_reason", ""),
            "tool_calls": msg.get("tool_calls") or [],
            "usage": resp.get("usage", {}) or {},
            "elapsed_s": elapsed,
            "ttft_s": elapsed,
            "raw": resp,
        }
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("bad response shape: " + json.dumps(resp)[:500])


def last_answer(content, reasoning):
    """Text to grade: the visible content, falling back to reasoning when the
    engine folded the answer there (some stacks never split reasoning out)."""
    c = (content or "").strip()
    return c if c else (reasoning or "").strip()


def final_number(text):
    """Last integer/float token in the answer (thinking models restate numbers;
    the final one is the answer)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


# ---------------------------------------------------------------- report
class Result:
    """Accumulates named sub-test outcomes; prints PASS/FAIL lines + summary."""

    def __init__(self, name):
        self.name = name
        self.cases = []

    def add(self, test, ok, detail=""):
        self.cases.append({"test": test, "ok": bool(ok), "detail": str(detail)[:400]})
        print(("PASS  " if ok else "FAIL  ") + test + ("  | " + str(detail)[:160] if detail else ""))
        return bool(ok)

    @property
    def n_pass(self):
        return sum(1 for c in self.cases if c["ok"])

    @property
    def ok(self):
        return self.cases and all(c["ok"] for c in self.cases)

    def finish(self, args=None, extra=None):
        summary = {
            "suite": self.name,
            "ok": self.ok,
            "passed": self.n_pass,
            "total": len(self.cases),
            "cases": self.cases,
        }
        if args is not None:
            summary["base_url"] = getattr(args, "base_url", None)
            summary["model"] = getattr(args, "model", None)
        if extra:
            summary.update(extra)
        print(f"\n{'PASS' if self.ok else 'FAIL'}  {self.name}: {self.n_pass}/{len(self.cases)}")
        if getattr(args, "save_result", None):
            with open(args.save_result, "w") as f:
                json.dump(summary, f, indent=2)
            print("saved: " + args.save_result)
        raise SystemExit(0 if self.ok else 1)
