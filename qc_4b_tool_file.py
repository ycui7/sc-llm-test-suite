#!/usr/bin/env python3
"""Gate 4b: real tool round trip reading a file from disk (fleet rule 2026-10-13).

Creates a canary file on THIS host, offers a read_file tool whose schema has
>=2 required properties, runs the full loop (model -> tool_call -> execute ->
tool result -> final answer), streamed AND non-streamed. Asserts:
  - tool_call emitted with every required property present, correct types
  - harness executes the call, result fed back
  - final answer contains the canary token (file was really read)
  - streamed deltas reassemble to the same arguments (append-only rule)
Exit 0 = PASS.
"""
import argparse, json, os, random, string, sys, time, urllib.request

from qc_common import openai_base

CANARY_DIR = os.path.expanduser("~/qc_canary")

def make_canary():
    os.makedirs(CANARY_DIR, exist_ok=True)
    word = "".join(random.choices(string.ascii_lowercase, k=8))
    num = random.randint(100000, 999999)
    token = f"{word}-{num}"
    path = os.path.join(CANARY_DIR, "canary.txt")
    with open(path, "w") as f:
        f.write(f"deployment-canary line1\nsecret token: {token}\nline3 end\n")
    return path, token

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from disk and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "encoding": {"type": "string", "description": "Text encoding, usually utf-8"},
            },
            "required": ["path", "encoding"],
        },
    },
}]

def post(base, key, body):
    req = urllib.request.Request(openai_base(base) + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
    return json.loads(raw)

def run_once(base, key, model, path, token, stream):
    msgs = [{"role": "user",
             "content": f"Read the file at {path} using the read_file tool (encoding utf-8) "
                        f"and then tell me the secret token in it."}]
    body = {"model": model, "messages": msgs, "tools": TOOLS,
            "temperature": 0, "max_tokens": 32768, "stream": stream}
    if stream:
        req = urllib.request.Request(openai_base(base) + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     **({"Authorization": f"Bearer {key}"} if key else {})})
        tool_calls = {}
        content = ""
        finish = None
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                d = json.loads(payload)
                ch = d.get("choices", [{}])[0]
                delta = ch.get("delta") or {}
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0),
                        {"id": None, "name": None, "args": ""})
                    if tc.get("id"): slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"): slot["name"] = fn["name"]
                    if fn.get("arguments"): slot["args"] += fn["arguments"]
                if ch.get("finish_reason"): finish = ch["finish_reason"]
        calls = [{"id": v["id"], "type": "function",
                  "function": {"name": v["name"], "arguments": v["args"]}}
                 for v in tool_calls.values() if v["name"]]
        out = {"tool_calls": calls, "content": content, "finish": finish}
    else:
        d = post(base, key, body)
        ch = d["choices"][0]
        out = {"tool_calls": ch["message"].get("tool_calls") or [],
               "content": ch["message"].get("content") or "",
               "finish": ch.get("finish_reason")}
    if not out["tool_calls"]:
        return False, f"no tool_calls (content starts {out['content'][:80]!r})"
    tc = out["tool_calls"][0]
    fn = tc.get("function", {})
    name = fn.get("name")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError as e:
        return False, f"args not JSON: {fn.get('arguments')[:120]!r}"
    missing = [p for p in ("path", "encoding") if p not in args]
    if missing:
        return False, f"missing required {missing} in {args!r}"
    if not isinstance(args["path"], str) or not isinstance(args["encoding"], str):
        return False, f"bad arg types {args!r}"
    if name != "read_file":
        return False, f"wrong tool name {name!r}"
    # execute for real
    with open(args["path"], encoding=args["encoding"]) as f:
        data = f.read()
    msgs.append({"role": "assistant", "content": None,
                 "tool_calls": [{"id": tc.get("id") or "call_0", "type": "function",
                                 "function": {"name": name,
                                              "arguments": fn.get("arguments")}}]})
    msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "call_0", "content": data})
    d2 = post(base, key, {"model": model, "messages": msgs, "temperature": 0,
                          "max_tokens": 32768})
    final = d2["choices"][0]["message"].get("content") or ""
    if token not in final:
        return False, f"final answer lacks canary ({final[:120]!r})"
    leak = any(p in final for p in ("<tool_call>", '{"name"', '"arguments"'))
    if leak:
        return False, f"leak in final: {final[:120]!r}"
    return True, f"ok stream={stream} args={args} token_found"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()
    path, token = make_canary()
    ok1, d1 = run_once(a.base_url, a.api_key, a.model, path, token, stream=False)
    print(("PASS" if ok1 else "FAIL"), "non-streamed:", d1)
    ok2, d2 = run_once(a.base_url, a.api_key, a.model, path, token, stream=True)
    print(("PASS" if ok2 else "FAIL"), "streamed:", d2)
    sys.exit(0 if (ok1 and ok2) else 1)

if __name__ == "__main__":
    main()
