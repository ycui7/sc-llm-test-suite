# sc-llm-test-suite

A rudimentary collection of basic test scripts to qualify a local model
serving. Zero third-party dependencies (Python 3 stdlib only); talks to any
OpenAI-compatible endpoint (LiteLLM, vLLM, sglang, llama-server).

## Qualification routine

Each routine is a standalone script. All share the same CLI:

```
--base-url   default http://localhost:4000/v1   (env QC_BASE_URL)
--model      default: first id from /models
--api-key    default from env QC_API_KEY
--max-tokens default 32768                     (thinking models empty
             `content` on small budgets — never lower this below 4096)
--save-result  write JSON summary to a path
```

| # | Script | What it checks | Pass bar |
|---|--------|----------------|----------|
| 1 | `test_endpoint.py` | `/models` up, model registered, minimal chat answers | 3/3 |
| 2 | `test_coherence.py` | deterministic temp-0 tasks: arithmetic, word problem, syllogism (no overreach), JSON copy, primes, instruction following, safety refusal, self-consistency x3, 150-word generation | all green |
| 3 | `test_tool_leak.py` | plain question *with tools in schema* → prose, no tool-JSON/prompt-echo leak; required tool → well-formed call; tool result consumed | 3/3 |
| 4 | `test_multiturn.py` | agentic loop: single-tool offset compose, 3-turn context persistence, two-tool arithmetic difference (deterministic fake backends), plus **C4: three tool-calling rounds in one conversation with `chat_template_kwargs {enable_thinking, preserve_thinking}` on and the thinking block replayed as `reasoning_content` on rounds 2 and 3** | 6/6 |
| 5 | `test_needles.py` | needles (unique word + random number, pinned seed) UNIFORMLY embedded in a public-domain-novel story haystack (`corpus/`, see `fetch_corpus.sh`), distributed from the start up to min(2/3 ctx, 512K); output room reserved (loud abort if it cannot fit); ALL needles must return in ONE inference; ctx<=64K -> 1024 needles, else 2048 | 2048/2048 (1024/1024 at ctx<=64K) |
| 6 | `test_speed.py` | pp2048 / tg512 @ concurrency 1,2,4 via llama-benchy (fallback: stdlib round-robin, directional only) | table recorded |

`test_smoke.sh` is the minimal bash ping (endpoint + PONG) kept for quick checks.

## Full run

```bash
python3 run_all.py --model ds4f-gb10 --ctx 1048576
# quality-only (skip the speed bench):
python3 run_all.py --model ds4f-gb10 --ctx 1048576 --skip speed
```

`run_all.py` runs the checks in gate order, writes per-check JSON plus
`verdict.json` to `results/<timestamp>/`, and exits 0 only if everything
passes (`VERIFIED`) — any FAIL means the deployment is NOT verified.

## House rules baked into these scripts

- Probes run at `max_tokens=32768` (4K hard floor): reasoning models burn the
  budget in thinking and return empty content on tiny caps.
- Needles are random-unique (unique word + unique number, no shared words,
  pinned seed) — ungamable by pattern completion.
- Bench at 3/4 of served context, cold page cache on the serving node, and
  zero interference during speed runs (launch, wait for exit, then read).
- Prefer going through LiteLLM for app-path traffic; benchmark the raw
  engine endpoint only when the raw numbers are what's under test.
- `test_multiturn.py` C4 talks to the **engine directly**: `chat_template_kwargs`
  (`enable_thinking`, `preserve_thinking`) are not forwarded by the gateway on
  plain chat, so the preserve-thinking round trip can only be qualified on the
  engine endpoint. Flags: `--no-preserve-thinking` (control run, thinking is not
  replayed), `--allow-no-reasoning` (waive C4c for a non-thinking model),
  `--think-kwargs '<json>'` (template-kwarg override).
