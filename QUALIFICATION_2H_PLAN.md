# 2-Hour Model Qualification Plan (come & go gates)

**Objective:** decide `VERIFIED` / `NOT VERIFIED` for a model lane in ≤120 min wall,
with every gate scored mechanically and every failure classified
(`infra` / `grader bug` / `config artifact` / `real model failure`) before it is reported.

**Harness:** `sc-llm-test-suite` (stdlib-only, no agent-tooling deps) +
`llama-benchy` + `lm-eval-harness` (scoring only). All traffic through Bifrost
(`http://sc-hub:4000/v1`); the raw engine endpoint is used **only** for the speed matrix.

---

## 1. Gate → coverage map (user's required dimensions)

| # | Required dimension | Gate | Mechanism | Pass bar |
|---|---|---|---|---|
| 1 | inference speed | `test_speed.py` (benchy) | pp2048 / tg512 @ conc 1,2,4, 3 runs | table recorded, no collapse, ttfr sane |
| 2 | coherent | `test_coherence.py` | temp-0 deterministic: arith, word problem, syllogism (no overreach), JSON copy, primes, instruction-follow, safety refusal, self-consistency ×3, 150-word | **100%** (this is the corruption canary — a quant/kernel bug fails *here*, not on GSM8K) |
| 3 | needle retrieval | `test_needles.py` | random unique word+5-digit, uniform strata, pinned seed, one-shot, 3/4 ctx | 2048/2048, `finish:stop`, `lines_emitted == N` |
| 4 | dense retrieval | `test_dense_retrieval.py` **(new)** | top-k over a many-short-record corpus + compose/aggregation queries | ≥90% top-1, ≥95% top-3 |
| 5 | tool calling | `test_tool_leak.py` T2/T3 | schema-valid required call, args, tool-result consumption | 100% schema-valid; ≥2/3 rounds |
| 6 | tool-call leaks | `test_tool_leak.py` T1 | plain question with tools in schema | 0 tool-JSON / prompt-echo / CoT leak in `content` |
| 7 | multi-turn tool call | `test_multiturn.py` | C1 offset-compose, C2 3-turn continuity, C3 two-tool arithmetic (deterministic fake backends) | **3/3** |
| 8 | breadth of knowledge | `test_knowledge.py` **(new)** | MMLU-Pro 30 (seeded) + commonsense 20 (TruthfulQA-style) | ≥ model's own published avg −8pp, and no category ≤30% |
| 9 | logic reasoning | `test_logic.py` **(new)** | 30 locked items: K&N knights/knaves, set/syllogistic, probability, causal, constraint puzzle | ≥80% |
| 10 | GSM8K / HS / college | `test_knowledge.py` **(new)** | GSM8K 50 seeded (HS) + MATH 25 level-3..5 (college) via lm-eval, exact-match | GSM8K ≥90% & ≥pub−5pp; MATH ≥ pub−10pp |
| 11 | computer use | `test_computer_use.py` **(new)** | 8-step deterministic fake-OS env over the **tools API** (obs→click→type→submit) | ≥6/8 goal states, 0 hallucinated actions, 0 loops |
| 12 | long-horizon coding | `test_coding_long.py` **(new)** | 3 tasks, multi-turn: T1 patch→test→pass, T2 debug failing suite, T3 build CLI to spec (tests provided) | ≥2/3 pass; all code exec'd, never read-graded |

Seeds are pinned for every battery → re-derivable A/B across lanes. Item banks live in
`items/` (JSONL), versioned in git.

---

## 2. Budget model — this is the whole trick

The 2-hour ceiling is a **token-budget problem**, not a number-of-tests problem.
Run the speed matrix *first*, get `tg_tps` / `pp_tps`, then compute the tier:

```
wall ≈ 25 min fixed (preflight, graders, report, per-scenario retry allowance)
     + (gen_tokens  / tg_tps) / 60
     + (prompt_tokens / pp_tps) / 60
```

Measured demand per tier (T1 baseline): **≈130K generated tokens**, **≈860K prompt
tokens** (needles haystack + dense-retrieval corpus + coding turns dominate the prompt side).

| Lane class | tg_tps | T1 | T1+T2 | T1+T2+T3 |
|---|---|---|---|---|
| Desktop GPU (~250 t/s, this box, measured 257/460/648) | 150–300 | ~35 min | ~55 min | ~85 min ✅ fits 2h |
| Cluster / mid (~40 t/s) | 40 | ~65 min | ~85 min | ~115 min — tight |
| Edge (~8 t/s, Orin-class) | 8 | >2h | — | — ✗ T1 only, reduced needles, say so |

**Auto-triage rule:** `wall_est > 100 min` → drop T3, then shrink the 2048 gate to 1025.
`wall_est > 180 min` → T1 only + reduced needle count, **and label the record as a
reduced gate** (never silently lower the bar on a reasoning-capable model).

---

## 3. Schedule (T1 core, ~40 min measured on this box's lane)

| t+ | Min | Gate | Note |
|---|---|---|---|
| 00:00 | 5 | **Preflight** | `GET /v1/models` via Bifrost, exact served id, `/health`, engine log clean, **drop_caches** on serving node, gateway datasheet gate (context_length + input_modalities + non-empty pricing). Fast-fail 400s: server-rejects-oversized (no auto-truncate), tool parser enabled, ctx ≥ 128K. |
| 00:05 | 4 | Coherence | temp 0, thinking OFF for output-only tasks, `max_tokens=32768` always |
| 00:09 | 3 | Tool-call leak + tool calling | 3/3 |
| 00:12 | 5 | Multi-turn tool call | C1/C2/C3 — **the 20-min gate**, hard timeout 15 min |
| 00:17 | 8 | Knowledge battery | GSM8K 50 + MATH 25 + MMLU-Pro 30 + commonsense 20, thinking OFF, parallel-safe (independent requests, no shared slot) |
| 00:25 | 10 | Logic battery | 30 locked items, thinking ON, 32K budget |
| 00:35 | 6 | Computer use | 8 actions, tool-API fake OS |
| 00:41 | 8 | Long-horizon coding | 3 tasks, real pytest runs |
| 00:49 | 8 | Dense retrieval | short-record corpus, top-k + compose |
| 00:57 | 10 | **Speed (benchy)** | last of the quality-neutral block, **zero interference** |
| 01:07 | 6 | Needle gate 256 @ 3/4 ctx | uniform, one-shot |
| 01:13 | ~8 | **Needle gate 2048 @ 3/4 ctx** | measured **320 s** on this lane |
| 01:21 | 6 | Report | histogram + verdict table + record md |
| 01:27 | — | **slack for 1 retry per scenario** | |

**Hard rule: the speed matrix and the needle gates never run concurrently** — each
needs an idle endpoint (zero-interference policy). Everything else may overlap.

---

## 4. Tiers

- **T1 — Must-have (no ship without it).** Gates 1–7 + GSM8K-50. A quant/kernel/context
  bug on the come&go surface is *coherence, needle, tool-leak, multi-turn* — gates 2,3,6,7.
  These are the highest-signal-per-minute gates and they're what breaks in practice.
- **T2 — Discriminating quality (do it if the budget model allows).** Gates 8–12 + the
  2048-needle gate. Answer "is this the model I think it is" (knowledge/reasoning) and
  "is it agentic enough to point an agent at".
- **T3 — Only when the lane is slow-enough budget.** Saturation/queueing (200% of KV
  pool, judge steady class only), long-context self-termination probe at full ctx,
  instruction-following stress (multi-constraint, format-locked), multilingual,
  vision (generated-PNG 4-case gate) if the weights are multimodal.

**Explicitly OUT (no time, low ROI):** full 12K-sample GSM8K/MMLU (±0.5 pts of noise at
n=30; it's a fingerprint test, not a benchmark), MMLU-Pro full, HumanEval/MBPP as a score
(1 turn each, subsumed by gate 12), LiveCodeBench, GPQA, MRCR/LongBench (my needle gate
is stricter at exactly the depth you care about), SWE-bench (hours), OSWorld (real VM,
hours), MT-Bench/Arena-Hard (needs a judge model = interference + cost), 128K-needle,
spec-decode acceptance sweeps, per-quant ladders, and all infra gates.

---

## 5. Scoring rules that keep it honest

1. **Exact-match / regex graders only. No LLM judge.** An LLM judge costs a second
   model, adds interference, and is unauditable.
2. **Normalized numeric extraction** (U+2212 → `-`, strip thousands separators,
   `answer is X` anchored) *before* extraction — Qwen-class models emit unicode minus
   and a naive `-?\d+` false-FAILs a correct `−10`.
3. Grade **`content` else `reasoning_content`**; recall/enum probes run thinking OFF so
   the answer lands in content. `finish:length` + empty content = **budget failure, not
   a retrieval failure**.
4. **Save raw replies + the item table** next to every result JSON (mandatory on first
   run of each gate). A score-only artifact gets rejected — you audit the raw text, and
   discarding replies forces a multi-hour rerun.
5. **Every FAIL gets a one-line classification** with evidence (server error body /
   regraded raw answer): `infra` | `grader bug` | `config artifact` | `real miss`.
   Verify the checker's expected value against the fake backend table before trusting
   a FAIL.
6. **Reasoning-config artifact check before reporting any FAIL:** a thinking model at
   `reasoning_effort=high` burns the whole budget on meta-deliberation and emits zero
   answer lines. Re-run **once** at effort=low / thinking-off, then call it a model
   failure.
7. **Prompt sizing measured, never guessed:** fixed-point calibrate chars/token on the
   *needle-bearing mix* via a `max_tokens=8` probe; pre-flight assert
   `est_prompt + max_tokens ≤ served ctx`. A sizing bug that 400s the first request is
   a harness FAIL that masquerades as a model FAIL.
8. **Prefix-cache honesty:** salt each request's prompt front with a unique fingerprint
   for concurrency/saturation tests and assert `cached_tokens == 0`. Report
   `usage.prompt_tokens_details.cached_tokens` on the deep-context run — a 0% cache
   report with a big TTFT drop is a *reporting* bug, not absence of caching.
9. **Needle-density confound:** when sweeping context length, hold density constant
   (scale N with ctx) or label the curve density-confounded. Uniform-per-band miss =
   density; 100%-then-0% tail = depth cliff. Classify before reporting.
10. Client per-request timeout ≥3000 s (deep prefill); fire one warmup request through
    the same route first (cold-boot JIT/kernel storm eats request #1).

---

## 6. Gates: pass / warn / fail

| Gate | PASS | WARN | FAIL |
|---|---|---|---|
| Coherence | 100% | — | any miss (self-consistency ×3 differs → corruption) |
| Needle 2048 @3/4 | 100%, `stop`, `lines==N` | ≥99.5% | <99.5%, `length`, `lines>N` (repetition), filler-shaped answers (coherence fail) |
| Speed | table + agg ≥0.8× baseline | ≥0.6× | <0.6× baseline, or ttfr ≫ baseline (spec-decode draft stall) |
| Tool leak | 0 leaks | CoT visible w/ reasoning off | tool-JSON / prompt echo in content |
| Tool calling | 3/3 schema-valid | 2/3 | malformed args, fabricated tool result |
| Multi-turn | 3/3 | C2 only, `tool_calls==0` → parser config | context loss across turns |
| GSM8K-50 | ≥90% & ≥pub−5pp | ≥pub−10pp | <pub−10pp |
| MATH-25 | ≥pub−10pp | — | <pub−20pp |
| MMLU-Pro-30 | ≥pub−8pp | ≥pub−15pp | a category ≤30% (dead-weight signature) |
| Logic-30 | ≥80% | 70% | <70% (overreach: answers yes/no where the answer is *undetermined*) |
| Computer use | ≥6/8 | 5/8 | hallucinated observation, >2 repeat loops |
| Long-horizon coding | ≥2/3 tasks green | 1/3 | any fabrication (passes by faking files) |
| Dense retrieval | top-1 ≥90% | 80% | verbatim-string wins, compose loses |

Baselines: the model's official published numbers, or the previously-qualified lane at
identical args (strict A/B — flip one variable).

---

## 7. Ops rules baked in

- **One endpoint at a time**, sequential, **zero interference** during speed/needle
  runs: launch in background, never poll mid-run, report on exit. One owner per lane.
- **Cold page cache** is the baseline (`drop_caches` on the serving node before each
  gate group). Warm cache inflates speed and needle numbers.
- `max_tokens=32768` for every probe (4K hard floor — reasoning models return
  `content:null` on tiny caps).
- **Per-scenario budget:** 15 min hard timeout, 1 retry, then record as FAIL-with-reason.
  The 2048-needle gate is exempt from the 15-min cap (measured 320 s; allow 20 min).
- Kill and retire the previous chain before relaunching — never two chains on one lane.
- **Secrets:** runners read the key from a chmod-600 file (`~/.config/qc/apikey`) and
  pass `--api-key "$AK"`; never put key literals in tool-written scripts (the scrubber
  turns headers into `***` and the run 401s looking like an auth bug).
- Raw artifacts to a **persistent** dir (`~/qc_results/<ts>/`), never `/tmp` — a
  recovery reboot wipes `/tmp` and the evidence is gone.
- Record markdown **is** the deliverable: no record → no "verified".

---

## 8. Verdict policy (two lanes, not one)

`run_all.py` prints **TIER1: PASS/FAIL** and **TIER2: PASS/FAIL** separately, plus
`VERIFIED` / `VERIFIED-WITH-CAVEATS` / `NOT VERIFIED`:

- Any T1 gate fail → **NOT VERIFIED** (fix, re-run the failing gates only).
- T1 pass, T2 fail → **NOT VERIFIED** for agentic duty, publishable as
  *chat-qualified / not-agent-qualified*.
- T1 pass + T2 warn → **VERIFIED-WITH-CAVEATS** (record lists the caveats verbatim).

Per-dimension routing: multi-turn/tool fail = not agentic; coherence/needle fail =
not long-context; logic/needle fail = not RAG. A lane can be fit-for-one-purpose.

---

## 9. Build order (in the sanctioned repo)

1. `test_knowledge.py` — seeded item banks, lm-eval scoring via
   `lm_eval --model local-chat-completions` (config-only, no new client code),
   `--num_fewshot 0` (few-shot is a prompt-injection / extra-token hazard), exact-match
   normalization, ±5% CI print, `--limit`-style 50/25 sampling with pinned seed.
2. `test_logic.py` — 30 locked items + normalized graders (the `unknown` trap item).
3. `test_dense_retrieval.py` — import `build_haystack`/`gen_needles` from
   `test_needles.py`; many-short-record corpus + compose queries.
4. `test_computer_use.py` — fake-OS env over the tools API (reuse
   `test_multiturn.py`'s deterministic backend loop).
5. `test_coding_long.py` — repo scaffolds in `items/coding/{t1,t2,t3}/`, pytest runner,
   graded on **real** test exit codes, 30-turn / 12-min caps.
6. Extend `run_all.py`: T1/T2 tiering, per-scenario wall caps, tier-aware verdict.
7. `qc_report.py`: per-dimension table + verdict; `qc_hist.py` misses-only histogram.
8. Run it on the current lane and save the baseline record — every future number is
   Δ% against this.

## 10. Standing risks to keep watching

- **Multi-turn continuity** — C2 3-turn context persistence is the #1 recurring failure
  on quantized MoE lanes (and it FAILed on the last run of this lane; re-check it first).
- **Long-enumeration degradation** — self-termination cliff (`stop` early, tail band 0%)
  and repetition loops (`lines_emitted > N`). Both are coherence, not retrieval.
- **Spec-decode side effects** — tg up, ttfr/e2e_ttft inflated (draft-block wait);
  report both columns, never tg alone.
- **Thinking verbosity runaway** — Qwen-class can burn 2000+ thinking tokens on an
  11-digit add; keep thinking OFF on output-focused tasks.
- **YaRN over-extension collapse** — past `n_ctx_train` at 3/4 ctx: thousands of extra
  needle-shaped tokens, low unique count, loop onset ~1/3 in. Model/extension failure.
