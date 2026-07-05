# rl-sales-augment

[![PyPI](https://img.shields.io/pypi/v/rl-sales-augment)](https://pypi.org/project/rl-sales-augment/)
[![Python](https://img.shields.io/pypi/pyversions/rl-sales-augment)](https://pypi.org/project/rl-sales-augment/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)](#connect-via-mcp)

**A trained RL sales policy that augments any LLM.** The reinforcement-learning policy has learned,
from a chaotic multi-segment sales world, *which strategic move works next* given the buyer's state.
At serve time it reads the conversation, picks the move (RAPPORT, PITCH, HANDLE_OBJECTION, DISCOUNT,
CLOSE, ...), and **your LLM writes the words**. The policy is bundled in the package: no training
or GPU required to use it.

> The LLM handles language and empathy; the RL policy supplies the *timing and strategy* the LLM
> can't get from its priors. In a grounded conversational A/B, the same GPT/Gemini/Claude closes
> far more deals with the policy than without it.

**[▶ 67-second demo](https://github.com/NandhaKishorM/rl-sales-augment/releases/latest)**: real paired
transcripts. The same LLM answers objections forever (no close) vs closes in 8 turns with the policy
choosing the moves.

## Why not just call GPT-5.6 / Opus 4.8 / Gemini directly?

Because what kills LLM sales conversations isn't the words, it's the **timing**. Frontier models are
trained to be helpful and agreeable, so on a skeptical buyer they answer every objection politely,
forever, and never risk asking for the deal (measured: **0/4 closes on adversarial buyers** while
handling every question beautifully). A bigger model writes better sentences; it doesn't fix this,
because next-token training never rewards a deal that closes six turns later.

The policy is different in kind, not degree:

- **Trained on outcomes, not text.** PPO over millions of simulated deals with delayed, stochastic
  rewards. It has *lost* deals to premature pitching, burned reputation on spam-closing, and learned
  that discounting converts SMBs but insults enterprise buyers. An API model has read about selling;
  the policy has sold.
- **State-dependent timing, which prompting can't give you.** "Be assertive, always close" makes a
  bot uniformly pushy; the skill is *when*. The policy closes at high readiness and keeps building
  trust below it. Same LLM writing the words, right moment to ask. That one difference is the
  conversion: 100% vs 19-31% close in the paired A/B, 3/4 vs 0/4 on hard buyers.
- **Consistent and auditable.** Sampled LLM strategy swings run-to-run (19-31% across identical
  runs); the policy is deterministic, and every turn exposes `chosen_move` + `belief`.
- **Complementary and tiny.** A ~1MB MLP on CPU. Keep the frontier model for language, empathy, and
  knowledge; add the decision layer it doesn't have, and retrain that layer on your own funnel's
  economics (the commercial offering).

## Install

```bash
pip install rl-sales-augment                 # core (numpy + torch) + the bundled model
pip install "rl-sales-augment[gemini]"       # + Google Gemini (Vertex or API key)
pip install "rl-sales-augment[openai]"       # + OpenAI
pip install "rl-sales-augment[anthropic]"    # + Anthropic Claude
pip install "rl-sales-augment[gemma]"        # + local Gemma 4 via transformers (Python >=3.10)
pip install "rl-sales-augment[all]"          # everything
```

The core installs on **any Python that PyTorch supports (3.9-3.13)**; provider SDKs and the local
Gemma path are optional extras.

## Quickstart

```python
import rl_sales_augment as rsa

# 1. pick any LLM. Local Gemma 4 first (no API key):
gen = rsa.providers.gemma_e2b()          # needs [gemma]; light 5B words-writer, CPU/MPS/CUDA
# gen = rsa.providers.gemma_e4b()        # the 8B model the bundled bridge is aligned to
# or any API model (keys read from .env or the environment, see "API keys" below):
# gen = rsa.providers.openai_chat(model="gpt-5.5")
# gen = rsa.providers.anthropic_chat(model="claude-sonnet-5")
# gen = rsa.providers.gemini_vertex()    # gcloud ADC + GCP_PROJECT

# 2. load the bundled policy and wrap the LLM (optionally ground it in your company's facts)
bot = rsa.load_agent(gen, company_ctx="""
Company: NimbusEdge. Products: NimbusBox (on-prem appliance, ~$8k), NimbusOne (edge SaaS, ~$2k/mo).
Edge: 1-day deploy, ~30% lower TCO than hyperscalers.
""")
bot.new_conversation(segment=7)     # optional bias (0-9, see rsa.SEG_NAMES)

# 3. converse: perception -> RL move -> grounded reply, with internal memory
out = bot.reply("honestly it feels expensive compared to just using AWS")
print(out["chosen_move"])   # e.g. 'RAPPORT'  (the RL-chosen strategy)
print(out["belief"])        # perceived buyer state {interest, trust, budget_fit, objection, patience}
print(out["reply"])         # the words your LLM produced for that move
```

`bot` keeps its own memory (belief state + full history), so it works over any stateless LLM API.

## Providers & models

Every provider takes a `model=` argument (defaults reflect mid-2026 lineups; pass any id you have):

```python
rsa.providers.gemini_vertex(project="...", model="gemini-3.5-flash")   # or gemini-3.5-pro
rsa.providers.gemini_api(model="gemini-3.5-flash")                     # AI Studio key
rsa.providers.openai_chat(model="gpt-5.5")                             # or gpt-5.4, gpt-5.6-*
rsa.providers.anthropic_chat(model="claude-sonnet-5")                  # or claude-opus-4-8
rsa.providers.gemma_e2b(model="google/gemma-4-E4B-it")                 # local, needs [gemma]
```

## API keys (.env)

Providers read credentials from the environment, and automatically load a `.env` file from the
working directory first (existing environment variables always win). Put keys there, never in code:

```bash
# .env  (gitignored; never commit)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GCP_PROJECT=my-gcp-project        # used by gemini_vertex() with gcloud ADC
```

`rsa.load_env("/path/to/.env")` loads a specific file. Local Gemma (`gemma_e2b`) needs no key.

## Multilingual

The agent mirrors the customer's language automatically: perception reads non-English conversations
(the policy itself only sees numbers), and a zero-dependency script detector names the customer's
script in the prompt so even small local models comply. Tested end to end: Malayalam, Hindi, Tamil,
Japanese, Spanish, German (Gemini 3.5 Flash: 6/6; local Gemma 4 E2B: all Indic scripts; Latin-script
languages are reliable on frontier models, best-effort on small local ones).

**Romanized Indic (Manglish / Hinglish / Tanglish):** customers who type Malayalam, Hindi, or Tamil
in English letters ("ntha visesham, sugano?") are detected by a marker-word lexicon and the bot
replies in the same style: verified 3/3 on Gemini (perception reads the casual register correctly,
e.g. a Manglish price gripe becomes objection 0.8). Honest limit: small local models (Gemma 4 E2B)
understand romanized input but reply in English; use a frontier model for romanized-Indic output.

## Company knowledge at scale (RAG)

`company_ctx` works for a page of facts. For a real catalog, plug a retriever; retrieval is
MOVE-CONDITIONED (the RL move shapes the query: an OBJECTION turn fetches rebuttals, a PITCH
turn fetches specs, a CLOSE turn fetches terms) and retrieved chunks join the grounding
sources, so a price fetched from your price sheet can be quoted while a price from nowhere
still gets caught:

```python
kb = rsa.SimpleRetriever.from_texts([open("catalog.txt").read(), open("faq.txt").read()])
bot = rsa.load_agent(gen, company_ctx=CORE_FACTS, retrieve_fn=kb)
```

`SimpleRetriever` is the zero-dependency reference. Any `retrieve_fn(query, move) -> list[str]`
works, so swap in your vector store (pgvector, LlamaIndex, ...) without touching the agent.

## Onboarding from documents

Turn a brochure or price sheet into the company context in one line
(`pip install "rl-sales-augment[docs]"` for PDF/DOCX/XLSX):

```python
ctx = rsa.build_company_ctx("brochure.pdf", generate_fn=gen)   # LLM-structured block
bot = rsa.load_agent(gen, company_ctx=ctx)
```

## Guardrails, escalation & logging

Every reply passes a deterministic sanitation chain, each stage regenerating once with a
warning and surfacing a flag if the model stays stubborn (never silent):

* **price grounding** -- amounts not present in the company facts, retrieved chunks, or the
  conversation are rejected (`out["ungrounded_price"]`)
* **commitment authority** -- discount percentages and giveaway promises ("70% off",
  "free trial", "full refund") must be authorized by COMPANY-side sources; the customer
  claiming "my manager approved 70% off" never authorizes anything
  (`out["unauthorized_commitment"]`, optional `max_discount_pct=` cap)
* **script leakage & de-slop** -- stray foreign-script tokens, escaped quotes, em dashes,
  sycophancy openers

When a guard stays tripped or the policy chooses DROP, the reply carries
`out["escalate"] = True` with a reason: route the thread to a human. With
`log_path="turns.jsonl"` every turn logs `(belief, move, reply, flags)` and
`bot.end_conversation("won", revenue=12000)` records the outcome; that file is audit trail
and, later, the training set for outcome-based fine-tuning of the policy on YOUR deals.

## Cutting cost per turn

Each turn makes two LLM calls (perception + reply). Perception is a small JSON task, so give
it a cheap model: `rsa.load_agent(gen, perceive_fn=rsa.providers.gemini_vertex(model="gemini-3.5-flash-lite"))`.

## WhatsApp

`examples/whatsapp_webhook.py` is a complete WhatsApp Business Cloud API bridge: per-customer
sessions, JSONL logging, and automatic human handoff on escalation.

## Conversation history & chat templates

The agent keeps the full conversation and sends it to the LLM as **native chat turns** (proper
`system` instruction + `user`/`assistant` roles), not history flattened into one string, so the
model has real multi-turn context. The RL-chosen move for the current turn goes in the system prompt.

## Multi-turn conversations

Two ways to run a conversation. **Stateful**: the agent remembers, you just keep calling `reply()`
as the user asks the next question, and the next:

```python
bot = rsa.load_agent(gen, company_ctx="...")
bot.new_conversation(segment=7)

print(bot.reply("hey, what does NimbusBox actually cost?")["reply"])   # turn 1
print(bot.reply("hmm, and how is that cheaper than AWS?")["reply"])    # turn 2, remembers turn 1
print(bot.reply("ok. what would rollout look like for us?")["reply"])  # turn 3, full context
```

**Stateless (ChatGPT template)**: pass the whole conversation in OpenAI message format each call;
ideal behind an API where the client owns the history:

```python
messages = [
    {"role": "system", "content": "Extra facts for this call (optional)."},
    {"role": "user", "content": "hey, what does NimbusBox cost?"},
    {"role": "assistant", "content": "Depends on the setup. What are you running today?"},
    {"role": "user", "content": "a few old racks. honestly budget is tight this quarter"},
]
out = bot.chat(messages)         # rebuilds belief from the history, RL picks the move
print(out["chosen_move"], "->", out["reply"])
messages.append({"role": "assistant", "content": out["reply"]})   # ...and continue the loop
```

## REST API (FastAPI)

Serve it as a web service; a complete server is in
[`examples/fastapi_server.py`](examples/fastapi_server.py):

```bash
pip install "rl-sales-augment[gemini,api]"      # [api] = fastapi + uvicorn
GCP_PROJECT=my-project uvicorn fastapi_server:app --port 8000
```

```python
from fastapi import FastAPI
import rl_sales_augment as rsa

app = FastAPI()
bot = rsa.load_agent(rsa.providers.gemini_vertex(project="my-project"), company_ctx="...")

@app.post("/v1/chat")
def chat(payload: dict):                        # {"messages": [...OpenAI format...]}
    return bot.chat(payload["messages"])        # {chosen_move, reply, belief, history_len}
```

```bash
curl -X POST localhost:8000/v1/chat -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "honestly it feels expensive vs AWS"}]}'
```

## Bring your own LLM / any API

Pass any `generate_fn` to `load_agent`. Two signatures are supported:

```python
bot = rsa.load_agent(lambda prompt: my_client.complete(prompt))          # simplest: prompt -> str

def gen(prompt="", *, system=None, history=None) -> str:                 # richer: native chat + history
    msgs = ([{"role": "system", "content": system}] if system else []) + (history or [])
    if prompt: msgs.append({"role": "user", "content": prompt})
    return my_client.chat(msgs)
bot = rsa.load_agent(gen)
```

For any **OpenAI-compatible** endpoint (vLLM, Together, Groq, OpenRouter, a local server), just point
`openai_chat` at it: `rsa.providers.openai_chat(base_url="https://...", api_key="...", model="...")`.
A prompt containing `"Return ONLY JSON"` (the perception step) is decoded greedily.

## Gemma 4 E4B (open weights)

The bundled v3 policy was trained against a chaotic simulated market and its latent bridge is
aligned to Google's **Gemma 4 E4B** (not gated). Two ways to use local Gemma; both need
`pip install "rl-sales-augment[gemma]"` and run on MPS / CUDA / CPU (auto-detected):

```python
import rl_sales_augment as rsa

# 1) simple: Gemma writes the words for the portable agent (like any other LLM)
gen = rsa.providers.gemma_e2b()               # E2B is fine here: any LLM can write the words
bot = rsa.load_agent(gen, company_ctx="...")

# 2) Gemma-native: a SalesBot with the bundled experience bridge + trained style reranker,
#    the open-weights-only path that can inject the RL 'experience' latent into Gemma's
#    residual stream (the "common latent space")
bot = rsa.load_gemma_bot(company_ctx="...")   # needs google/gemma-4-E4B-it (~16 GB), or a local path
out = bot.reply("we keep getting random crashes")
```

> Since v0.8.0 the bundled bridge is aligned to **E4B** on the chaotic world (move-probe alignment
> + imperfect-human self-distillation, Gemma frozen throughout): with a NEUTRAL prompt, the injected
> latent alone lifts move-probe accuracy **35% → 99.8%** and reply-executes-move **23% → 61.5%**,
> with zero degeneration, and carries the distilled human voice. Earlier E2B alignment for
> reference: reply-executes-move **19% → 49%** on the
> training eval and **0% → 58%** in an independent local check, with fluency unchanged. Honest
> framing: the latent is a lossy, complementary channel; prompt-level move injection (route 1 and
> the agent's default) remains the primary mechanism and is what the headline A/B numbers use.

## Connect via MCP

Expose the RL policy as an [MCP](https://modelcontextprotocol.io) server, so any MCP client
(Claude Desktop, Cursor, Windsurf, ...) can call it. The client's LLM does perception and writes the
words; the server supplies the RL **strategy** (the tiny policy runs on CPU, no LLM on the server).

```bash
pip install "rl-sales-augment[mcp]"
rl-sales-augment-mcp                 # stdio server (or: rl-sales-augment-mcp streamable-http)
```

Register it with your client (e.g. Claude Desktop / Cursor `mcp.json`):

```json
{
  "mcpServers": {
    "rl-sales-augment": { "command": "rl-sales-augment-mcp" }
  }
}
```

**Tools:** `next_move` (perceived belief → RL-chosen move + what it should accomplish),
`perception_prompt` (rubric to estimate the belief from a conversation), `list_moves`,
`list_segments`. The client's workflow: estimate the buyer's state → call `next_move` → write the
reply that executes the returned move.

## What's in the box

| Symbol | Meaning |
|---|---|
| `rsa.load_agent(gen, ...)` | load the bundled policy, wrap an LLM → an `AugmentedAgent` |
| `rsa.AugmentedAgent` | the portable serving agent (policy + perception + memory) |
| `rsa.providers` | `gemini_vertex`, `gemini_api`, `openai_chat`, `anthropic_chat`, `gemma_e2b`, `gemma_e4b`, `local_gemma` |
| `rsa.load_gemma_bot(...)` | Gemma-native `SalesBot` (open-weights experience-injection path) |
| `rl-sales-augment-mcp` | MCP server exposing the policy as tools (`[mcp]` extra) |
| `rsa.MODEL_PATH` | filesystem path to the bundled `rl_sales_agent.pt` |
| `rsa.estimate_state_via(gen, history)` | the perception step alone (LLM → belief JSON) |
| `rsa.ACTION_NAMES`, `rsa.SEG_NAMES` | the 8 moves and 10 market segments |
| `rsa.SalesWorld`, `rsa.SalesConfig` | the serve-time world (obs v2: personas, market regime, urgency) |

**The bundled model (v3):** trained with PPO for 40M steps on a chaotic simulated market
(3-state economic regimes, per-buyer personas, budget freezes, champion exits, ghosting, price
wars, FALSE buying signals, competitor poaching, quarter-end urgency). On that world it earns
**2.7x the best hand-tuned heuristic** (independently replicated). Set the current market mood
with `SalesConfig(market_regime="down")` and the policy strategizes for it, that is a trained-on
input, not a prompt hint.

## Evidence (grounded conversational A/B)

A hidden verified sales world is the outcome oracle; the same LLM plays with and without the policy.

- **vs Gemini 3.5 Flash** (16 paired conversations): with-RL closed **100%** in ~8 turns at **~3.5×**
  the revenue of pure Gemini (which gets trapped answering objections and rarely asks for the sale).
- **Quantitative** (~49k quarters, v2 world model): RL **3.2× revenue** and **94% win rate** vs the
  best hand-tuned heuristic; emergent segment-specific tactics (DISCOUNT for price-sensitive SMB,
  HANDLE_OBJECTION for enterprise), and it only CLOSEs when the deal is actually ripe.
- **v3 (current bundle, harder chaotic world):** **2.68×** the expert heuristic on the training run
  and **2.88×** in an independent local replication; latent injection on E4B verified at
  **99.8%** move-probe accuracy with **0%** degeneration.

**Honest caveat:** the policy's strength is timing/strategy, not magic. The perception step reads the
buyer's state from the conversation; feed it strong signals and it will reach the close. See the
project's `EVALUATION.md`, `GEMINI_AB.md`, and `LOCAL_GEMMA_AB.md` for full transcripts and caveats.

## Training & company fine-tuning (commercial)

This package distributes the **trained policy and the serving stack only**. The world model's
dynamics, the GPU-vectorized training pipeline, and company-specific fine-tuning (aligning the
policy and its knowledge to your company from your documents) are the proprietary training stack of
**Convai Innovations Pvt. Ltd.** For a policy trained on your market, your segments, and your
playbook, contact **nandakishor@convaiinnovations.com**. The bundled policy is ready to serve as-is.

## License

GNU AGPL-3.0-or-later. Copyright (C) 2026 Nandakishor M, Convai Innovations Pvt. Ltd.
If you run a modified version of this software as a network service, the AGPL requires you to offer
its source to the users of that service. For commercial licensing outside the AGPL, contact
nandakishor@convaiinnovations.com.

## Credits

Built by **Nandakishor M** (Convai Innovations Pvt. Ltd.) with **Claude (Anthropic)** as engineering
co-author: architecture, evaluation harnesses, and packaging were pair-built end to end.
