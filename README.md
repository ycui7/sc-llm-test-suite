# sc-llm-test-suite

A rudimentary collection of basic test scripts to qualify a local model
serving. Zero third-party dependencies (Python 3 stdlib only); talks to any
OpenAI-compatible endpoint (Bifrost, vLLM, sglang, llama-server).

## Qualification routine

Each routine is a standalone script. All share the same CLI:

```
--base-url   REQUIRED via env QC_BASE_URL or the flag — the fleet gateway
             URL is deliberately not committed
--model      default: first id from /models
--api-key    default from env QC_API_KEY (never hardcode keys in scripts —
             read them from a chmod-600 file, e.g. ~/.config/qc/apikey)
--max-tokens default 32768                     (thinking models empty
             `content` on small budgets — never lower this below 4096)
--save-result  write JSON summary to a path
```

`run_all.py` runs the numbered checks in gate order (see below).

| # | Script | What it checks | Pass bar |
|---|--------|----------------|----------|
| 1 | `test_endpoint.py` | `/models` up, model registered, minimal chat answers | 3/3 |
| 2 | `test_coherence.py` | deterministic temp-0 tasks: arithmetic, word problem, syllogism (no overreach), JSON copy, primes, instruction following, safety refusal, self-consistency x3, 150-word generation | all green |
| 3 | `test_tool_leak.py` | plain question *with tools in schema* → prose, no tool-JSON/prompt-echo leak; required tool → well-formed call; tool result consumed | 3/3 |
| 4 | `test_multiturn.py` | agentic loop: single-tool offset compose, 3-turn context persistence, two-tool arithmetic difference (deterministic fake backends), plus **C4: three tool-calling rounds in one conversation with `chat_template_kwargs {enable_thinking, preserve_thinking}` on and the thinking block replayed as `reasoning_content` on rounds 2 and 3** | 6/6 |
| 5 | `test_prefix_cache.py` | prefix caching really works: one fresh-nonce prompt sent 3x, req2 TTFT < req1/2 + cached-token evidence (`usage.prompt_tokens_details.cached_tokens`; counter gate skipped — TTFT delta carries the verdict — when the engine's parser reports details as null, e.g. DS4F). Engines with no prefix cache FAIL | PASS |
| 6 | `test_needles.py` | needles (unique word + random number, pinned seed) UNIFORMLY embedded in a public-domain-novel story haystack (`corpus/`, see `fetch_corpus.sh`), distributed from the start up to min(2/3 ctx, 512K); output room reserved (loud abort if it cannot fit); chars/token calibrated from a low bootstrap to a fixed point; ALL needles must return in ONE inference; ctx<=64K -> 1024 needles, else 2048; misses draw the `qc_hist.py` box chart; `--reasoning-effort` passthrough for engines that ignore `chat_template_kwargs` | 2048/2048 (1024/1024 at ctx<=64K) |
| 7 | `test_speed.py` | pp2048 / tg512 @ concurrency 1,2,4 via llama-benchy (discovered via `QC_BENCHY` / `~/.venv/bin` / PATH; fallback: stdlib round-robin, directional only) **plus the concurrency-scaling gate (2026-11-26): aggregate tg at c2 must EXCEED 1.5x c1, else the endpoint serializes → bench aborts, FAIL** | table recorded + gate PASS |
| 8 | `test_table_alignment.py` | **INFORMATIONAL, no pass/fail**: model generates one ASCII table (unequal column widths, unique variable-length cell content); the probe counts table lines whose delimiter positions deviate from the majority pattern and reports the misaligned count (plus rows, columns, column widths, duplicate cells) | count reported, never gates |

## Standalone gates and helpers (not in `run_all.py` order)

| Script | Purpose |
|--------|---------|
| `qc_4b_tool_file.py` | **Gate 4b** (fleet rule 2026-10-13): real tool round trip reading a canary file from disk — model → tool_call → execute → tool result → final answer; asserts every required schema property is present with correct types, streamed deltas reassemble to the same arguments. Single-call smoke tests pass while real harness schemas fail — this gate catches that |
| `test_ssd_restore.py` | SSD KV-restore coherence gate (GB10/LMCache): cold phase → engine restart → warm phase must show ttft_cold/ttft_warm >= 10, warm answer identical to cold (temp-0), needle recall exact. Run `--phase cold`, restart engine, `--phase warm --expect <cold.json>` |
| `grade_needles_offline.py` | regrade a saved needle answer three ways (strict / tolerant / presence) so a FAIL is attributable without rerunning |
| `qc_coverage.py` | from a real run's `needles.json` + answer: per-depth-bin placed-vs-found chart proving a 2048/2048 came from a genuinely full-context pass |
| `qc_hist.py` | box-drawn missed-needles histogram (spec: `REQUIREMENTS_histogram.md`); `--demo` for the synthetic render |
| `qc_stream.py` | streaming chat helper with ttft/pp/tg telemetry + SSE idle-stall watchdog |
| `battery_glm53_exl3.sh` | example full battery for one lane (all traffic via the gateway, zero interference); key read from `~/.config/qc/apikey`, gateway from `QC_BASE_URL` |
| `test_smoke.sh` | minimal bash ping (endpoint + PONG) for quick checks |

## Full run

```bash
export QC_BASE_URL=https://<gateway-host>/v1   # fleet Bifrost gateway (not committed)
python3 run_all.py --model ds4f-gb10 --ctx 1048576
# quality-only (skip the speed bench):
python3 run_all.py --model ds4f-gb10 --ctx 1048576 --skip speed
```

`run_all.py` runs the checks in gate order, writes per-check JSON plus
`verdict.json` to `results/<timestamp>/`, and exits 0 only if everything
passes (`VERIFIED`) — any FAIL means the deployment is NOT verified.
`table-alignment` is reported as INFO and never gates the verdict.

## House rules baked into these scripts

- ALL AI traffic goes through the fleet Bifrost gateway (the gateway URL is
  deliberately not committed here — supply it via `QC_BASE_URL` or
  `--base-url`; LiteLLM retired 2026-09-25); benchmark the raw engine
  endpoint only when the raw numbers are what's under test.
- Probes run at `max_tokens=32768` (4K hard floor): reasoning models burn the
  budget in thinking and return empty content on tiny caps.
- API keys live in chmod-600 files (`~/.config/qc/apikey`), never in scripts;
  pass `--api-key "$(cat ~/.config/qc/apikey)"`.
- Needles are random-unique (unique word + unique number, no shared words,
  pinned seed) — ungamable by pattern completion.
- Bench at the tested depth (needles: min(2/3 ctx, 512K)), cold page cache on
  the serving node (GB10/gx10 nodes: `drop-caches` immediately before launch),
  and zero interference during speed runs (launch, wait for exit, then read).
- A tool-qualification PASS requires the real file round trip (gate 4b), not
  just a schema-valid single call.
- `test_multiturn.py` C4 talks to the **engine directly**: `chat_template_kwargs`
  (`enable_thinking`, `preserve_thinking`) are not forwarded by the gateway on
  plain chat, so the preserve-thinking round trip can only be qualified on the
  engine endpoint. Flags: `--no-preserve-thinking` (control run, thinking is not
  replayed), `--allow-no-reasoning` (waive C4c for a non-thinking model),
  `--think-kwargs '<json>'` (template-kwarg override).
