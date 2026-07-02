# SalesLLM-style benchmark

A third-party-*style* evaluation of the RL policy, modeled on the published methodology of
**"Sell More, Play Less: Benchmarking LLM Realistic Selling Skill"** ([arXiv 2604.07054](https://arxiv.org/abs/2604.07054),
the SalesLLM benchmark). It answers: *does the RL move-selection improve an LLM's selling skill on a
standardized, benchmark-shaped test — not just our own training world?*

## Why this harness (and what it is / isn't)
The official SalesLLM code, its `CustomerLM` simulator, and its BERT buying-intent classifiers were
**not released** as of mid-2026, so this is **not** the official benchmark and the numbers are not
comparable to its leaderboard. Instead it faithfully reproduces the paper's *method*:

- **Multi-turn persuasive dialogue** with realistic deal progression and a purchasing outcome.
- **CustomerLM-style buyer simulator** — an LLM held strictly in the BUYER role (their fix for the
  ~17% "role reversal" problem), with a hidden private disposition that only yields buying signals
  when the rep earns them.
- **Controllable personas × difficulty** — cooperative (`easy`) → ROI-skeptical (`medium`) →
  adversarial/price-focused (`hard`), matching the paper's "calibrated customer profiles".
- **Automatic evaluation pipeline** — an LLM rater for **sales-process progress** (0–1 across
  rapport → discovery → value → objection-handling → close) plus an end-of-dialogue **buying-intent**
  judge (their BERT classifier → an LLM judge here).

## The comparison
The **same base LLM** (Gemini 3.5 Flash, via `rl-sales-augment`'s provider) is the seller in both
arms, paired on identical scenarios/personas. The only difference is who picks the strategic move:

- **base LLM** — the model sells on its own priors.
- **+ RL policy (ours)** — the trained policy picks the move; the LLM writes the words.

Scenarios span SalesLLM's domains (Financial Services, Consumer Goods) plus B2B tech (dev tool,
retail hardware).

## Run it
```bash
pip install "rl-sales-augment[gemini]" matplotlib
python benchmarks/salesllm_style_eval.py 7 --project YOUR_GCP_PROJECT
# -> salesllm_results.json (raw numbers + transcripts) and salesllm_results.png (plot)
```

## Results

Gemini 3.5 Flash seller, 12 paired dialogues (4 scenarios × 3 difficulties), 7 turns each. See
`salesllm_results.png` (plot) and `salesllm_results.json` (raw numbers + full transcripts).

**Buying-intent conversion (buy rate):**

| Difficulty | base LLM | + RL policy |
|---|---|---|
| easy (cooperative) | **100%** | 75% |
| medium (ROI-skeptical) | **100%** | 50% |
| hard (adversarial / price-focused) | **0%** | **75%** |
| **overall** | **67%** | **67%** |

Sales-process progress (0–1): overall base 0.76 vs RL 0.72, but on hard buyers RL leads 0.75 vs 0.65.

**Interpretation (honest).** Overall it's a **tie** — and *where* each wins is the real finding:
- On **easy/medium** buyers a strong modern LLM already closes on its own; the RL policy's
  discipline (build rapport, don't rush the pitch) sometimes *holds back* when an eager buyer would
  have bought immediately, so it slightly underperforms.
- On **hard, adversarial** buyers — the skeptical, price-focused CFOs who *don't* buy from a naive
  pitch — the base LLM converts **0/4**, while the RL policy's strategic sequencing (rapport →
  objection-handling → timed close) converts **3/4**.

So the RL move-selection matters **exactly where selling is hard**. That's a more credible and useful
claim than a blanket multiplier: *a good LLM suffices on easy deals; the learned strategy is what
rescues the tough ones.* (Small sample — treat as directional; increase scenarios/turns and repeat.)

## Honest caveats
- Simulated buyers (like the paper, and like our own eval) — this validates the **method** against a
  standardized shape, a step up from our bespoke world, but it is still not real-customer revenue.
- Small sample (a dozen paired dialogues per run) — increase scenarios/personas/turns and repeat for
  tighter estimates. Buyer + judge are stochastic (an LLM), so expect run-to-run variance.
- The judge is an LLM, not the paper's fine-tuned BERT; treat scores as relative (base vs RL), not
  absolute or leaderboard-comparable.
