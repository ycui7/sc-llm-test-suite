"""Streaming chat completion with prefill/decode rate telemetry.

Stdlib only. Sends stream=true and measures:
  ttft_s   request sent -> first content/reasoning token  (prefill time)
  total_s  request sent -> stream end
  pp_tps   prompt_tokens / ttft_s          (prefill rate)
  tg_tps   completion_tokens / (total-ttft) (decode rate)

If the server rejects stream_options, retry without it and count SSE chunks
(directional token proxy). A per-chunk idle-stall watchdog raises if no SSE
chunk arrives within idle_stall_s seconds. Extra kwargs (e.g.
chat_template_kwargs) are merged into the request payload.
"""
import json
import time
import urllib.error
import urllib.request


def _open(url, api_key, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


class StreamStall(Exception):
    """No SSE chunk arrived within the idle-stall window."""


def stream_chat(base_url, api_key, model, messages, max_tokens=32768,
                timeout=3600.0, temperature=0.0, idle_stall_s=300.0,
                extra=None, **kwargs):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "stream": True, "stream_options": {"include_usage": True}}
    if extra:
        payload.update(extra)
    payload.update(kwargs)
    # socket read timeout sits just above the idle window so a dead stream
    # surfaces as a caught URLError and the watchdog classifies it.
    sock_timeout = min(timeout, max(idle_stall_s + 10.0, 60.0))
    t0 = time.monotonic()
    try:
        resp = _open(url, api_key, payload, sock_timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        if "stream_options" in body:
            payload.pop("stream_options", None)
            resp = _open(url, api_key, payload, sock_timeout)
        else:
            raise
    # ---- SSE parse loop ----
    content = []
    reasoning = []
    usage = {}
    ttft = None
    n_chunks = 0
    finish = None
    last_chunk_t = t0
    DONE = '[' + 'DONE' + ']'
    it = iter(resp)
    while True:
        now = time.monotonic()
        if now - t0 > timeout:
            resp.close()
            raise StreamStall('overall timeout %.0fs exceeded' % timeout)
        if now - last_chunk_t > idle_stall_s:
            resp.close()
            raise StreamStall('no SSE chunk in %.0fs' % idle_stall_s)
        try:
            raw = next(it)
        except StopIteration:
            break
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            resp.close()
            if time.monotonic() - last_chunk_t > idle_stall_s:
                raise StreamStall('stall: ' + repr(e)[:120])
            raise
        # a keep-alive body returns empty b'' at EOF instead of raising
        # StopIteration; treat it (and blank lines) as end-of-stream.
        if not raw:
            break
        line = raw.decode(errors='replace').strip()
        if not line:
            continue
        if not line.startswith('data:'):
            continue
        last_chunk_t = time.monotonic()
        data = line[5:].strip()
        if data == DONE:
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if obj.get('usage'):
            usage = obj['usage']
        ch = obj.get('choices') or []
        if not ch:
            continue
        n_chunks += 1
        d = ch[0].get('delta') or {}
        if ch[0].get('finish_reason'):
            finish = ch[0]['finish_reason']
        c = d.get('content')
        rc = d.get('reasoning_content')
        if c:
            content.append(c)
        if rc:
            reasoning.append(rc)
        if (c or rc) and ttft is None:
            ttft = time.monotonic() - t0
    total_s = time.monotonic() - t0
    if ttft is None:
        ttft = total_s
    pt = usage.get('prompt_tokens') or 0
    ct = usage.get('completion_tokens') or 0
    pp_tps = (pt / ttft) if (ttft and pt) else None
    decode_t = total_s - ttft
    tg_tps = (ct / decode_t) if (decode_t > 0 and ct) else None
    return {
        'content': ''.join(content),
        'reasoning': ''.join(reasoning),
        'finish': finish,
        'usage': usage,
        'ttft_s': ttft,
        'total_s': total_s,
        'pp_tps': pp_tps,
        'tg_tps': tg_tps,
        'n_chunks': n_chunks,
    }
