# pm1 evaluation

This document describes how pm1 was evaluated and reports each experiment: what was tested, the rule fixed before the
run, the result with its 95% interval, and whether the change shipped. The headline tables are collected in
[RESULTS.md](RESULTS.md); the system itself is described in [DESIGN.md](DESIGN.md).

Negative results are reported with the same care as positive ones. Most ideas did not pass their pre-registered rule.

## Contents

- [Method](#method)
- [LoCoMo: pm1 against other memory systems](#locomo-pm1-against-other-memory-systems) (E1–E15)
- [Retrieval benchmarks: LongMemEval-S, coding trajectories, CPU-only](#retrieval-benchmarks) (E16–E18)
- [BEAM-1M: long conversations](#beam-1m-long-conversations) (E19–E25)
- [Latency](#latency)
- [Error analysis: where the remaining errors come from](#error-analysis-where-the-remaining-errors-come-from)
- [Open items](#open-items)

## Method

### Benchmarks

| Benchmark | Size used | What it tests | Metric |
|---|---|---|---|
| **LoCoMo** (LoCoMo-10) | 10 conversations, 272 sessions, 5,882 turns; **1,540 questions** (categories 1–4: multi-hop 282, temporal 321, open-domain 96, single-hop 841; adversarial category 5 dropped, as Mem0 and others do) | answering questions about long chats | judged accuracy |
| **LongMemEval-S** (cleaned) | **470** answerable questions (30 abstention questions excluded) | finding the right sessions | `strict_all@5`: all gold sessions among the top 5 distinct sessions |
| **BEAM-1M** | dev split: conversations 1–20, **400 questions** (10 types × 40); test split (conversations 21–35, 300 questions) kept sealed | very long conversations (about 1,700–3,000 messages each) | official rubric, 0 / 0.5 / 1 per item; event ordering scored by aligned Kendall tau |
| **REALTALK** | **728** questions | real (not synthetic) long chats | judged accuracy |
| **Coding trajectories** | 40 public SWE-agent trajectories in one shared store, 88 programmatic questions, 85 paraphrased versions | "which session changed file X / hit error Y / handled issue Z" | hit@1 on the target session |
| **LifeGen 1M** | synthetic personal chats, English and Chinese | memory updates and follow-ups | accuracy (LifeGen's scorer) |
| **PerLTQA** | 1,200 Chinese questions | Chinese retrieval guardrail | R@1, evidence within 2K tokens |

All data is public. Sources and licenses of the datasets used by the code in this repository are in
[THIRD_PARTY.md](THIRD_PARTY.md). Benchmarks are used for testing
only, never for training.

### Same conditions for every system

- **Same answer model and prompt.** On LoCoMo every system's context is answered by GPT-6-Luna (reasoning high) with
  Mem0's official LoCoMo answer prompt, unchanged. Using a competitor's prompt keeps the numbers comparable with Mem0's
  published ones and avoids tuning the prompt toward pm1. On BEAM, contexts are answered by Claude Haiku 5.5 with the
  official BEAM `answer_generation_for_rag` prompt (one series also tests Mem0's own answering rules).
- **Same budget rule.** Contexts are cut to 1,024 or 4,096 tokens (o200k_base tokenizer): items are added in the
  system's own order, items that do not fit are skipped, and the result is sorted by time. Systems that pack to a
  budget themselves are called once per budget.
- **Same write-time LLM.** Systems that call an LLM when they store memories (Mem0, Graphiti, Hindsight, OpenViking,
  graphify) all call the same model through one proxy, so their write costs are counted the same way. Each system
  uses its official default configuration; every deviation is documented in the run notes.
- **Blind judge.** Answers from all arms are shuffled (seed 2026), given neutral ids, and judged without the system
  name or budget. LoCoMo uses composer-2.5 with Mem0's lenient LoCoMo judge prompt (partial answers count, dates within
  14 days count, so absolute scores run high; comparisons are unaffected). BEAM uses GPT-6.1-Sol with the official
  rubric. The judge is never the answer model.
- **Judge and batch noise.** Re-judging the same answers moves a score by about 0.5 points; re-answering with the
  same prompt moves it by about 1 point on LoCoMo and more on REALTALK. Blind re-judging of 400 already-judged answers
  agreed 94–97% of the time across five checks. **Scores from different judges are never subtracted.**

### Statistics and decision rules

- **Paired bootstrap.** Every comparison resamples questions, paired across arms, 10,000 times (seed 2026) and reports
  the mean difference and its 95% interval. Where an arm was answered twice, each question's score is the mean of the
  two batches.
- **Pre-registration.** Each experiment's arms, data split, metric, primary comparison and decision rule were written
  down and committed before any answer was generated. Changes after seeing data were logged as numbered amendments,
  and anything chosen after looking at results is labeled exploratory.
- **Rules for an improvement.** A change counts as an improvement only if the 95% interval of its pre-registered
  primary comparison lies above 0 and its guardrails hold (for example, no drop on the other benchmark or on Chinese
  retrieval). A change that passes is still shipped only after a separate decision; several passed and were not
  shipped because of cost or latency.
- **Rules against competitors.** Lower bound > 0: ahead. Upper bound < 0: behind. Whole interval within ±3 points: tie.
  Otherwise: inconclusive. The main LoCoMo comparison uses Holm correction over the competitors.
- **Two retrieval paths.** **[prod]** is `search_sessions`, the tool agents call. **[eval]** is `memory_recall`, the
  path most LoCoMo experiments used; it has extras production does not (neighbour turns from the top 5 hits, the
  optional fact index) and reranks only the first 16 candidates. Every row below says which path it measures.

---

## LoCoMo: pm1 against other memory systems

All LoCoMo rows: 1,540 questions, GPT-6-Luna answers with Mem0's prompt, composer-2.5 blind judge, unless noted.

### E1. First comparison: pm1 against Mem0, Hindsight, Graphiti, OpenViking, graphify

**Tested.** The original pm1 retrieval ([eval], capped at 12 records) against five systems at their default
configurations.

**Rule.** Primary budget 1,024 tokens; lead / tie / behind per the competitor rule, Holm-corrected over four
competitors.

**Result.**

| System | 1K | 4K | pm1 − system at 1K [95% CI] | Verdict |
|---|---|---|---|---|
| Mem0 2.2.1 | 0.892 | 0.906 | −0.099 [−0.121, −0.079] | pm1 behind |
| Hindsight 1.0.0 | 0.838 | 0.857 | −0.046 [−0.070, −0.021] | pm1 behind |
| pm1 (original) | 0.792 | 0.795 | | |
| Graphiti 0.31.2 | 0.699 | 0.716 | +0.093 [+0.066, +0.120] | pm1 ahead |
| graphify 0.9.74 | 0.527 | 0.560 | +0.266 [+0.236, +0.295] | pm1 ahead |
| OpenViking 0.5.2 | 0.455 | 0.474 | +0.337 [+0.307, +0.368] | pm1 ahead |

The 4K verdicts were the same. Write cost: pm1 0 LLM calls, Mem0 6,453, Graphiti 18,835, Hindsight 742, OpenViking
1,002, graphify 302.

**Error analysis.** Checking the gold evidence turns against pm1's context showed the problem was retrieval, not
answering: 88% of pm1's wrong answers at 1K (281 of 319) had evidence missing from the context. The packer stopped
early: at a 4K budget it used only 681 tokens, so evidence coverage barely moved (0.616 → 0.619). Multi-hop and
open-domain questions, whose evidence is spread over several sessions, had all evidence in context only 38–39% of the
time.

**Decision.** Fix the packer (E2).

### E2. Use the whole budget and add neighbour turns (pm1-v2)

**Tested.** Remove the 12-record cap so the packer fills the budget, and let the top 5 hits bring ±2 turns from their
session. The variant was chosen on conversations 0–4 with an offline evidence check and confirmed on conversations
5–9 (all-evidence share at 1K: 0.587 → 0.701).

**Rule.** pm1-v2 − pm1, interval above 0.

**Result.** 1K 0.838 (+0.046 [+0.030, +0.063]); 4K 0.894 (+0.099 [+0.080, +0.117]). Passed at both budgets. Still
LLM-free; retrieval p50 117 ms. The remaining gap to Mem0 at 1K is mostly multi-hop questions (0.734 vs. 0.890).

**Decision.** Shipped (the record cap and the neighbour turns are in both retrieval paths).

### E3. Fact index (pm1-v3)

**Tested.** One LLM call per session extracts self-contained facts, each pointing back to the turns it came from. At
query time, the turns behind the facts nearest the question join the candidates. Only raw turns go into the context;
facts serve as search keys. Write cost: 275 calls for the 272 sessions, about 1/23 of Mem0's 6,453.

**Rule.** pm1-v3 − pm1-v2, interval above 0.

**Result.** 1K 0.870 (+0.032 [+0.016, +0.048]); 4K **0.922** (+0.028 [+0.015, +0.042]). Passed. Against Mem0
(descriptive only, not the primary comparison): 1K −0.021 [−0.039, −0.004], 4K +0.016 [+0.003, +0.030].

A variant with facts written by a local Qwen3.5-4B (no external API) reached 0.846 / 0.911: about 60% of the gain at
4K (+1.7 vs. +2.8 points) and almost none at 1K.

**Decision.** Not shipped. The fact index needs an LLM at write time, which production avoids, and on BEAM-1M it
gave no gain (−0.9 points). It remains an [eval]-path option (`PM1_FACTS`).

### E4. Entity one-hop on top of facts (pm1-v4)

**Tested.** Add facts that share an entity with the question. Offline, it raised the 4K all-evidence share from
0.856 to 0.873 (multi-hop 0.738 → 0.788).

**Result.** 4K 0.924, v4 − v3 +0.002 [−0.008, +0.012]: a tie. The extra evidence did not change answers.

**Decision.** Not shipped (off by default as `PM1_FACT_HOP`).

### E5. Rerank every candidate and graph two-hop recall (pm1-v5d, pm1-v5)

**Tested.** Rerank all candidates instead of the first 16 (v5d), then add two-hop recall over the fact graph (v5).
Offline, the 4K all-evidence share on held-out questions rose from 0.755 to 0.877.

**Rule.** pm1-v5 − pm1-v3 at 4K above 0.

**Result.** v5 0.882 / 0.918; v5 − v3 at 4K −0.004 [−0.015, +0.007], at 1K +0.012 [−0.004, +0.029]. Not passed.
Retrieval latency rose 7–10× (p50 181 ms → 1,867 ms on this path).

**Decision.** Not shipped. More evidence in the context did not turn into more correct answers when accuracy was
already 0.92.

### E6. Porter stemming of the keyword index (pm1-v3p)

**Tested.** Porter stemming for the English keyword index. With BM25 alone it raised R@1 by 0.050 offline.

**Rule.** Primary comparison against a same-code re-run without stemming (pm1-v3r), because the code had moved on
since v3.

**Result.** 4K +0.001 [−0.009, +0.010] (30 wins, 29 losses): a tie. Vectors and the reranker already recover what
stemming adds. The same-code re-run itself differed from the original v3 by +0.012 [+0.002, +0.021] at 1K, a reminder
that a re-run moves scores by about 1 point.

**Decision.** No measurable effect. Stemming is enabled in the production index (since 2026-10-31); its LoCoMo effect
is this null result. pm1-v3r (0.882 / **0.929**) is the best LoCoMo row overall, but it is a re-run of the fact-index
variant, not a different system.

### E7. The production tool on LoCoMo (`search_sessions`)

**Tested.** The production path measured on the same questions, with the old v2, v3 and v5 answers re-judged in the
same batch. Arms: `k=20` (the tool's maximum) at 1K and 4K; the tool default at the time (`k=8`, 2K budget); and
`k=200` at 4K (not allowed by the tool, measurement only).

**Rule.** Six pre-registered comparisons of `k=20` against v2, v3 and v5 at both budgets.

**Result.**

| Arm | Budget | Context tokens used | All evidence in context | Accuracy |
|---|---|---|---|---|
| pm1-v5 [eval] | 4K | 3,446 | 90.1% | 92.3% |
| pm1-v3 [eval] | 4K | 3,434 | 85.8% | 92.1% |
| `search_sessions` k=200 [prod] | 4K | 2,148 | 81.0% | 90.1% |
| pm1-v2 [eval] | 4K | 3,422 | 81.5% | 89.9% |
| **`search_sessions` k=20 [prod]** | 4K | 1,282 | 78.6% | **89.4%** |
| `search_sessions` default k=8 [prod] | 2K | 683 | 74.5% | 88.0% |
| **`search_sessions` k=20 [prod]** | 1K | 522 | 71.3% | **86.4%** |
| pm1-v2 [eval] | 1K | 844 | 70.9% | 85.5% |

| Comparison | Difference [95% CI] | Verdict |
|---|---|---|
| prod k=20 − v2, 1K | +0.91 [−0.84, +2.66] | not separable |
| prod k=20 − v2, 4K | −0.52 [−2.01, +1.04] | not separable |
| prod k=20 − v3, 1K | −0.65 [−2.40, +1.10] | not separable |
| prod k=20 − v3, 4K | −2.79 [−4.35, −1.30] | production lower (no fact index) |
| prod k=20 − v5, 1K | −2.08 [−3.64, −0.58] | production lower |
| prod k=20 − v5, 4K | −2.99 [−4.55, −1.43] | production lower |
| prod k=200 − prod k=20, 4K | +0.71 [−0.65, +2.01] | not separable |

On REALTALK the production tool was 4.40 points below the v5 configuration at 4K (83.9 vs. 88.3, [−7.01, −1.65]).
At `k=20` and a 4K budget the tool used only about 1.3K tokens of text: LoCoMo turns are short and `k` limits the
number of hits.

**Decision.** Measurement only. The production numbers in [RESULTS.md](RESULTS.md) (89.4% at 4K, 86.4% at 1K) come
from this run. They were measured before later production changes (default budget raised from 2K to 4K, the next
turn of each hit, session layout), which were not re-measured with this judge.

### E8. How the LLM-free production path compares with Mem0

The production tool scored **89.4%** at 4K (E7) and Mem0 **90.6%** (E1): about 1 point below Mem0. At 1K the gap is
about 2.8 points (86.4% vs. 89.2%). These differences subtract two separate reports that were not judged in the same
batch, so there is no paired interval for them and they should not be read as a tie or as a significant difference.
Mem0 needed 6,453 LLM calls to ingest the corpus; pm1 needed none.

### E9. Does the reranker help answers?

**First test ([eval] path).** pm1-v3r with and without its reranker, 769 questions (a fixed half of the set, chosen by
question-id hash), GPT-6.1-Sol judge. Primary rule at 4K: interval above 0.

| Budget | Reranked | Not reranked | Difference [95% CI] |
|---|---|---|---|
| 4K (primary) | 0.940 | 0.939 | +0.001 [−0.010, +0.013] |
| 1K | 0.902 | 0.876 | **+0.026 [+0.008, +0.044]** |

Primary not passed. This path reranks only the first 16 fused candidates.

**Second test ([prod] path).** `search_sessions` reranking all 100 candidates, against reranking only the first 16 and
against no reranking. Same store, same code, 1,540 questions, Sol judge.

| Budget | Rerank 100 (production) | Rerank 16 | No rerank | 100 − none [95% CI] | 100 − 16 [95% CI] |
|---|---|---|---|---|---|
| 1K | **0.884** | 0.849 | 0.786 | **+9.8 [+8.0, +11.7]** | +3.5 [+2.1, +5.0] |
| 4K | **0.916** | 0.882 | 0.878 | **+3.8 [+2.5, +5.2]** | +3.4 [+2.1, +4.7] |

Passed at both budgets; every question type gained, multi-hop most (1K +14.5, 4K +8.5). Wins / losses against no
rerank: 189 / 38 at 1K, 84 / 25 at 4K. The first test's null result at 4K came from reranking only 16 candidates
(rerank 16 − none at 4K: +0.4 [−0.3, +1.1]). This run also covers the production changes made after E7. Its scores
come from a different judge (Sol) and cannot be compared with the composer-judged table.

**Other data.**

- LongMemEval-S retrieval: reranking adds +0.6 [−0.6, +1.9] over fusion, within noise (E16).
- Coding trajectories, paraphrased questions: +5.9 [+1.2, +11.8] hit@1 (E18).
- LifeGen 1M (synthetic): removing the reranker *raised* accuracy by +2.18 [+0.90, +3.49] (English and Chinese
  combined), almost all of it on implicit updates (English 0.126 → 0.549). In those questions the evidence is a pair of
  adjacent turns, and only the first names the person. The second turn is reachable only as a neighbour, and the
  reranker pushed the naming turn below the five hits that bring neighbours. This finding led to the "next turn"
  option, which is now in production.

**Decision.** The production reranker (all 100 candidates) stays. It is also the main cost: see [Latency](#latency).

### E10. Larger budgets (6K, 8K)

**Tested.** v5 [eval] retrieval at 6K and 8K against 4K, LoCoMo and REALTALK, two answer batches each.

**Result.** The all-evidence share rose 1.7–5.0 points, accuracy did not: 6K +0.73 [0.00, +1.46], 8K +0.22
[−0.53, +0.95] (both datasets combined). Not passed (a lower bound of exactly 0 does not count).

**Decision.** Not adopted.

### E11. Presenting the evidence differently (the answer step)

**Tested.** With the v5 4K context held fixed: gold evidence only (`oracle`, diagnostic), absolute dates written next
to relative ones (`time`), relevance order instead of time order (`rel`), top 8 listed first (`mark`), a 2K budget
(`fewer`), a two-step answer that first lists relevant lines (`twopass`), and letting the model ask for one more
search (`requery`).

**Rule.** Each arm − `base` (a fresh re-answer of the same prompts), interval above 0.

| Arm | Accuracy | − base [95% CI] |
|---|---|---|
| base | 93.2 | |
| oracle (gold evidence only, diagnostic) | 94.9 | +1.76 [+0.26, +3.12] |
| rel | 93.7 | +0.52 [−0.39, +1.43] |
| requery | 93.6 | +0.39 [−0.45, +1.23] |
| twopass | 93.4 | +0.19 [−0.78, +1.17] |
| mark | 93.1 | −0.13 [−1.10, +0.84] |
| time | 92.6 | −0.58 [−1.43, +0.26] |
| fewer | 92.5 | −0.65 [−1.62, +0.26] |

None passed. A side finding: word-overlap F1, the metric several memory papers report, mostly measures answer length.
`twopass` raised F1 by 6.65 points, 6.37 of them on questions both arms answered correctly, because its answers were
shorter.

**Decision.** No change to packing or answering.

### E12. Larger models in the retrieval stack

| Idea | Result | Decision |
|---|---|---|
| 1024-d vectors (Qwen3-Embedding-0.6B) instead of e5 384-d | two fresh batches, LoCoMo + REALTALK: +0.13 [−0.55, +0.82] | not adopted |
| 2560-d vectors (Qwen3-Embedding-4B) | failed the evidence gate (4K all-evidence +0.33 on LoCoMo, +0.83 on REALTALK; +1 required) | not answered, not adopted |
| e5 + a 1024-d model as two fused vector lanes | LoCoMo 4K +0.46 [−0.13, +1.04]; stopped before REALTALK finished | not adopted |
| Qwen3-Reranker-4B instead of 0.6B | 1K, 600 sampled questions, two fresh batches: +2.25 [+0.33, +4.25]; but the Chinese guardrail failed (PerLTQA R@1 0.803 → 0.692) | not adopted |
| cascade: 0.6B ranks all, 4B re-ranks the top 20 | 1K +1.58 [−0.50, +3.67]; at 2K −0.58 [−2.25, +1.00] | not adopted |

### E13. Smarter recall: second round, planning, graph

| Idea | Result | Decision |
|---|---|---|
| second-round recall from the top hits (pseudo-relevance feedback), reranked against the question | candidate pools grew, evidence in the context did not | not adopted |
| per-seed second-round lists fused with equal weight | 4K all-evidence share fell from 0.900 to 0.805–0.829; 1K fell much further | removed |
| plan the search first (an LLM lists aspects and sub-queries; local 4B or GPT-6-Luna as planner) | confirmation run: −0.08 [−1.17, +1.00] (LoCoMo + REALTALK) | closed |
| entity one-hop, graph two-hop (E4, E5) | LoCoMo evidence coverage up, accuracy unchanged | not adopted |
| literal (grep-style) match channel as a third fused list (CPU path) | LongMemEval-S `strict_all@5` 0.926 → 0.926; paraphrased trajectories hit@1 −3.4 points; latency 1 → 19 ms | not adopted |
| blending BM25 ranks into the final reranked order (`PM1_SESS_BLEND`) | no measured gain | off by default |

### E14. An expert grep agent as the retriever

**Tested.** Claude Haiku 5.5 with read-only shell tools (rg, grep, sed, …; at most 10 calls) searches each
conversation's text files and returns line ranges, which then go through the same budget cut, prompt and answer model.
Judged by Haiku (the composer judge was no longer available); a 100-question calibration found Haiku about 1 point
more lenient than composer, so grep's numbers are slightly flattering.

**Result.** 1K 0.845, 4K 0.847: tied with Hindsight (1K +0.007 [−0.016, +0.031]) and with pm1-v2 at 1K, below Mem0
(1K −0.046 [−0.066, −0.025]) and below pm1-v3 / v5 by 4–7 points. A bigger budget did not help, because the agent
already picks narrow ranges. Grep is weakest on temporal questions (category 2, 10–14 points behind), where clues
from several places must be combined. Cost: about 5.6 LLM tool calls and 8 seconds per question, against
milliseconds and no LLM for pm1's retrieval.

### E15. Published systems with different metrics: MemoryOS and MemSifter

**MemoryOS** (EMNLP 2025), run with its own evaluation code, same answer model and prompt, judged in one batch with
pm1-v5 and Mem0: **81.0** (context about 6.8K tokens) against pm1-v5 92.5 at 4K and Mem0 90.9 in that batch.
MemoryOS − pm1-v5 (4K) −11.58 [−13.54, −9.68]. Its retrieved pages held all gold evidence for only 39.2% of
questions (pm1-v5: 90.0%). It made 10,159 LLM calls to ingest. Its paper's high scores come from a very terse answer
prompt and an F1 metric: the same terse prompt raised pm1-v5's F1 by 21.3 points while its accuracy fell 2.9 points.

**MemSifter** reports word-overlap F1 on a 400-question LoCoMo subset. With the same F1 code, pm1-v5 scored 39.5 at
1K, against 41.8 in the paper (8K context, a different answer model) and 37.2 for Mem0 in our runs. A 75-question
run of the MemSifter ranking model itself did not improve pm1's answers (1K −5.3 [−13.3, +2.7] when it re-ordered
pm1's candidates, inconclusive) and was about 100× slower.

---

## Retrieval benchmarks

### E16. LongMemEval-S: pm1 against agentmemory

**Tested.** Retrieval only (no answer model): does each system put all gold sessions in its top 5 distinct sessions?
agentmemory was run with its own evaluation script, which reproduced its published BM25 (0.847) and grep (0.721)
numbers. pm1 indexes turn spans and folds them into sessions; agentmemory indexes whole sessions (each system's own
design).

**Rule.** pm1 production retrieval − agentmemory BM25, paired bootstrap; ±3 points is a tie.

**Result** (470 questions, `strict_all@5`):

| System | All | Dev (91) | Test (379) |
|---|---|---|---|
| agentmemory grep | 0.721 | 0.824 | 0.697 |
| agentmemory local vectors (MiniLM-L6) | 0.826 | | |
| pm1 keyword only | 0.834 | 0.857 | 0.828 |
| **agentmemory BM25** | 0.847 | 0.879 | 0.839 |
| pm1 vectors only | 0.887 | 0.912 | 0.881 |
| pm1 keyword + vectors (RRF) | 0.917 | 0.945 | 0.910 |
| **pm1 production (RRF + rerank)** | **0.923** | 0.945 | 0.918 |

- pm1 production − agentmemory BM25: **+7.7 [+5.1, +10.4]** (39 wins, 3 losses). Ahead.
- The lead comes from vectors and reranking: pm1's keyword search alone ties BM25 (−1.3 [−3.8, +1.1]).
- Biggest gains: multi-session (0.711 → 0.860), preference (0.833 → 0.967), temporal (0.780 → 0.858).
- agentmemory's local-vector path ran with 11 questions failing to start its sandbox; counted as misses it scores
  0.826, excluded 0.845.
- Its README reports R@5 95.2% on its own metric and split, which this run did not reproduce and is not comparable.

### E17. Coding-agent trajectories: pm1 against claude-mem and agentmemory

**Tested.** 40 public SWE-agent trajectories share one store. 88 questions are generated by program from structured
fields (which session modified a file, handled an issue, or hit an error line), each with one target session. Because
these questions quote the trajectory text, each was also paraphrased by Claude Haiku 5.5 from one detail only, with an
automatic check against copying four consecutive words or any file name; the paraphrased set is the primary result.
claude-mem's observer (Claude Haiku 5.5) watched each trajectory replayed one session at a time.

**Result** (hit@1):

| System | Paraphrased (85) | Original (88) |
|---|---|---|
| **pm1 production retrieval** | **0.976** | **0.977** |
| pm1 RRF (no rerank) | 0.918 | 0.977 |
| claude-mem | 0.859 | 0.807 |
| agentmemory BM25 | 0.847 | 0.955 |
| pm1 keyword only | 0.835 | 0.977 |
| agentmemory local vectors | 0.788 | 0.807 |
| grep | 0.541 | 0.773 |

Paraphrased: pm1 − claude-mem +0.118 [+0.059, +0.188] (10 wins, 0 losses); − agentmemory BM25 +0.129
[+0.059, +0.200]. On the original questions every keyword system is near the ceiling. Limits: one store, 88 questions
from the same sessions, so questions are not independent; results show direction, not size.

### E18. CPU-only path

**Tested.** What a user without a GPU gets: (C0) keyword only; (C1) the pip-installable path, e5 int8 ONNX on the CPU
plus keyword search, RRF, no reranker; (C2) the production e5 GGUF on a CPU llama.cpp server. Two CPU threads, no GPU
service. LongMemEval-S was run on a systematic sample of every fifth question (94 questions) because full indexing on
two threads would have taken about 15 hours; intervals are wider than on 470 questions.

**Rule.** C1 − agentmemory BM25 lower bound > 0 to claim "the CPU path still beats BM25"; otherwise report as tied or
lower.

**Result.**

| System | LongMemEval-S `strict_all@5` (94-question sample) | Trajectories, paraphrased hit@1 (85) |
|---|---|---|
| production full retrieval (GPU vectors + rerank) | 0.936 | 0.976 |
| **C1: CPU RRF, no reranker** | **0.926** | **0.918** |
| production RRF (no rerank) | 0.926 | 0.918 |
| agentmemory BM25 | 0.872 | 0.847 |
| claude-mem | — | 0.859 |
| C0 keyword only | 0.830 | 0.835 |

- LongMemEval: C1 − BM25 +5.3 [0.0, +10.6]; the lower bound is 0, so by the rule this is "higher or tied". C1 equals
  the GPU fusion stage (0.0 [−3.2, +3.2]).
- Trajectories: full retrieval − C1 +5.9 [+1.2, +11.8]; this is where the reranker helps. C1 − BM25 +7.1
  [+1.2, +14.1].
- Speed on about 745 spans, two threads, fresh queries: keyword 6 ms p50, vectors 5 ms, fused search 6 ms (p95
  8 ms). Building vectors: median 73 s per question's 700 spans; peak memory about 2.7–2.9 GB.

**Decision.** The CPU path is the default install. Not verified: CPUs without AVX512-VNNI, Chinese data on this path,
larger stores.

---

## BEAM-1M: long conversations

All BEAM rows: dev split, Claude Haiku 5.5 answers (two batches, averaged per question) unless noted, GPT-6.1-Sol
judge with the official rubric. Mem0's row uses Mem0's own published per-question answers (GPT-5 answering, its top
200 memories, its own answer prompt), re-judged by the same judge: 0.645 on 400 questions (Mem0 reports 0.641).

### E19. Production `search_sessions` at 6K against Mem0

**Tested.** The production tool at `k=20` and a 6,000-token budget, its JSON output placed in BEAM's official answer
prompt; and the older [eval] path at 4K.

**Result** (400 questions):

| Arm | Score |
|---|---|
| Mem0 (its own memories, prompt and GPT-5 answers) | **0.645** |
| pm1 production 6K | 0.535 |
| pm1 [eval] path 4K | 0.535 |
| pm1 production 6K answered by GPT-6-Luna | 0.469 |

Mem0 − pm1 production: **+11.0 [+7.0, +14.9]**. Mem0 leads on temporal reasoning (0.637 vs. 0.278),
multi-session reasoning, summarization, instructions and preferences; pm1 leads on abstention (0.709 vs. 0.556) and
contradiction resolution. The production context held all gold evidence for only 25.6% of the 348 questions with
gold turns (the [eval] path at 4K: 37.6%): whole records with JSON fields fit fewer distinct turns than short spans.
p50 latency per search was 4.8 s.

**Decision.** Measurement. This is pm1's largest measured weakness.

### E20. Plain-text layout, sub-queries, an LLM memory layer, more hits

**Tested.** Five changes against production 6K: plain dated text instead of JSON (`text`); the question split into
1–3 sub-queries by Haiku (`subq`); a Haiku-written memory layer (summaries and dated memory items per 60K-character
block, written without seeing any question; 34,484 items) replacing or joining the raw turns; and `k=50`.

**Result.** None differed significantly from production 6K:

| Arm | Score | − production 6K [95% CI] |
|---|---|---|
| LLM memory layer only | 0.560 | +2.48 [−0.73, +5.72] |
| raw turns + memory layer | 0.557 | +2.20 [−0.55, +4.91] |
| plain-text layout | 0.555 | +2.06 [−0.33, +4.42] |
| sub-queries | 0.542 | +0.73 [−1.87, +3.33] |
| k = 50 | 0.540 | +0.53 [−1.87, +2.94] |

Sub-queries and `k=50` raised evidence coverage, not scores. Mem0 still led the best arm by +8.5 [+4.9, +12.1].

### E21. A "list the dates first" answer prompt

**Tested.** Replace the official answer requirements with: list each fact the answer depends on as
`YYYY-MM-DD | fact`, compute durations and orders step by step, then give a final answer. Only the final answer is
judged. This is a non-standard prompt and is reported separately.

**Result.** Plain text + dates 0.582 (+2.68 [+0.44, +4.88] over the same context with the official prompt); memory
layer + dates 0.591 (+3.10 [+1.02, +5.13]). The biggest change was contradiction resolution (0.181 → 0.509), because
listing facts in time order puts conflicting statements side by side; summaries fell by about 10 points because
answers became shorter.

**Correction recorded during this series.** Mem0's 0.645 is not produced with BEAM's official short prompt. Mem0's
own evaluation answers with GPT-5, reads 200 dated memories, and uses a prompt that asks for accurate and complete
answers with nine rules (check all memories, prefer the newer of two conflicting facts, mind dates, and so on). Every
gap reported in E19–E21 therefore mixes memory, answer model and answer prompt.

### E22. Mem0's answering rules, and an event-row view written by a local 27B model

**Tested.** On conversations 11–20 (200 questions):

1. production 6K context with Mem0's answering rules instead of the official short prompt;
2. an *event-row view* added before the question: a locally served 27B model reads the user's messages once per
   conversation and writes dated event rows with quotes, which a program checks against the source; at question
   time the model classifies the question, and for ordering, contradiction and date-difference questions a program
   computes the answer from the rows and adds one checked fact with up to 10 dated quotes. Questions it does not
   handle get exactly the control prompt.

**Rule for the event rows.** On the three targeted types (60 questions) the interval is above 0; on the other seven
types the upper bound is ≥ 0; on abstention the mean drop is no more than 0.10.

**Result** (200 questions; the Mem0-rules arms combine two answer rounds, four batches per question):

| Arm | Score |
|---|---|
| production 6K, official prompt | 0.556 |
| + event rows | 0.591 |
| production 6K, Mem0's answering rules | 0.650 |
| **+ event rows** | **0.683** |
| Mem0 | 0.654 |

- Event rows on the official prompt: targeted types +0.102 [+0.039, +0.170], other types +0.005 [−0.007, +0.023],
  abstention unchanged. All three rules passed. A second answer round through a different API gave +0.057
  [−0.018, +0.133] on the targeted types (same direction, did not pass alone); the gain is somewhere between about
  +0.06 and +0.10.
- Mem0's answering rules alone brought production pm1 level with Mem0 (Mem0 − pm1 +0.004 [−0.056, +0.061]).
- With event rows added, Mem0 − pm1 was −0.029 [−0.084, +0.023]: pm1's mean was higher, the interval includes 0.

### E23. Is the lead over Mem0 real? (400 questions)

**Tested.** Build event rows for conversations 1–10 too (questions the rows had never seen) and combine with
conversations 11–20, with the merge rule fixed before any result.

**Rule.** pm1 − Mem0 lower bound > 0: lead shown. Mean > 0 with an interval including 0: lead not shown. No extra
batches or re-tuning afterwards.

**Result.**

| Arm | Conversations 1–10 | 11–20 | All 400 |
|---|---|---|---|
| pm1 6K + Mem0's rules + event rows | 0.641 | 0.683 | **0.662** |
| pm1 6K + Mem0's rules | 0.628 | 0.650 | 0.639 |
| pm1 6K, official prompt + event rows | 0.569 | 0.581 | 0.575 |
| pm1 6K, official prompt | 0.543 | 0.558 | 0.551 |
| Mem0 | 0.635 | 0.654 | **0.645** |

- pm1 − Mem0: **+0.018 [−0.019, +0.055]**. **Lead not shown.**
- The event-row rules replicated on the new conversations (targeted types +0.046 [+0.012, +0.083]).
- On the same context, switching from the official prompt to Mem0's rules moves pm1 from 0.551 to 0.639; the
  official prompt plus event rows is still 6.9 points below Mem0 (−0.069 [−0.107, −0.030]). Most of the BEAM gap is
  the answer step, not retrieval.
- Per type, pm1 leads on contradictions (0.655 vs. 0.241), abstention, knowledge updates and information extraction,
  and ties on temporal reasoning; Mem0 leads on preferences (0.855 vs. 0.771), summarization (0.661 vs. 0.504),
  multi-session reasoning, instructions and event ordering. Event rows are rarely triggered in those types.

**Decision.** Not shipped. Mem0's answering rules were used only to compare with Mem0. The event-row view needs a
27B model at write time (about 13 GPU-hours of model time for 10 conversations) and up to 20 model calls per question
(p90 92 s); that cost was not accepted for production.

### E24. A preference and standing-instruction lane

**Tested.** For every question, retrieve up to 8 stored user preferences and standing instructions by vector
similarity, let the 27B model judge each with a single yes/no token, and add up to 3 that apply.

**Rule.** Preference + instruction types (80 questions) above 0; abstention mean ≥ −0.025 with the section added to at
most 4 abstention questions; other types upper bound ≥ 0; all 400 above 0.

**Result.** Not passed: targeted types +0.006 [−0.021, +0.038]; all 400 +0.007 [−0.004, +0.019]. The section was added
to 293 of 400 questions, including 17 abstention questions, because "applies" was judged true for anything on the
same topic. On the official short prompt the section did help the targeted types (+0.056 [+0.006, +0.109]); Mem0's
rules already remind the model to respect user preferences.

**Decision.** Closed; no variants tuned on the dev set.

### E25. A cheap 2B pre-filter for the event-row indexer

**Tested.** Can a 2B model decide which user messages the 27B indexer needs to read? Required: at least 95% of the
key rows and of the gold-evidence messages kept, and at least 30% of tokens saved.

**Result.** No threshold met both. At P(yes) ≥ 0.3, 14.3% of tokens were skipped and 89.6% of key rows kept; keeping
both above 95% allows skipping only about 5% of tokens. The small model judged most messages worth reading (median
P(yes) 0.66).

**Decision.** Not adopted.

---

## Latency

| Setting | p50 | p95 / p90 |
|---|---|---|
| **production `search_sessions`, real calls over one week (291 calls)** | **3.7 s** | p95 7.0 s |
| production `search_sessions`, two weeks (565 calls) | 3.9 s | p95 8.1 s |
| production on BEAM-1M (about 1,700 turns per conversation) | 4.8 s | p90 10.0 s |
| CPU path, keyword + vector fusion, about 745 spans, two threads | 6 ms | p95 8 ms |
| LoCoMo [eval] path (one small store per conversation) | 150 ms | p95 431 ms |
| Mem0 retrieval on LoCoMo | 131 ms | p95 205 ms |
| Graphiti / OpenViking / graphify / Hindsight retrieval on LoCoMo | 1.2 s / 1.5 s / 4.5 s / 8.4 s | |

Almost all production latency is the reranker: Qwen3-Reranker-0.6B on vLLM scores 100 candidates, and the agent-facing
profile waits up to 8 s for it. The production store held 387,786 records and 808,087 spans, searched by exact vector
scan; the median call returned 3,814 tokens. The CPU path without a reranker answers in milliseconds but loses about
6 points on paraphrased trajectory questions (E18).

## Error analysis: where the remaining errors come from

The evidence points to the answer step, not retrieval, as the bottleneck once the budget is used well:

1. **Early on, retrieval was the problem.** In E1, 88% of wrong answers had evidence missing from the context. Using
   the whole budget and adding neighbour turns (E2) fixed most of that: +9.9 points at 4K.
2. **On LoCoMo the ceiling is near.** Gold evidence alone scores 0.949; the best pm1 row is 0.929. With the v5 context
   fixed, giving the model only the gold evidence adds 1.76 points (E11), and at 4K 83 of v5's wrong answers already
   had all gold evidence in context.
3. **More evidence did not mean more correct answers.** Entity hops (E4), graph recall (E5), larger budgets (E10),
   sub-queries and `k=50` on BEAM (E20) all raised evidence coverage without raising accuracy.
4. **On BEAM the answer prompt moves scores more than retrieval does.** With the same context, Mem0's answering rules
   add about 9 points (0.551 → 0.639, E23), the dates-first prompt about 3 (E21); no retrieval change in E20 was
   significant. Even when the production context held all gold evidence, the score was only 0.618.
5. **What retrieval of raw turns still misses** is aggregation over many sessions: BEAM summarization and
   multi-session questions, and LoCoMo multi-hop questions at 1K. Systems that merge facts at write time do better
   there.

## Open items

- **Sealed BEAM test.** A pre-registration for the 300 sealed test questions is drafted but not run. Its power
  analysis says that if the true lead is the +0.018 seen on dev, the test would show a lead only about 13% of the
  time; a non-inferiority result (no more than 5 points behind Mem0) is the likely outcome.
- **Cross-session aggregation.** The most promising direction is write-time consolidation by a small local model,
  added next to the raw records, tested on dev only with its own pre-registration.
- **Code-writing benefit.** A 52-task code benchmark showed no gain from pm1, but it cannot measure memory: the tasks
  are self-contained specifications, 52 tasks are too few (re-running one configuration moved it by 8.7 points), and
  only hidden tests are scored.
