# rl-sales-augment

A trained RL policy that picks the next sales move (rapport / pitch / objection / discount / close);
your LLM writes the words. The trained model ships in the package: CPU, no finetuning, works with
any LLM.

```bash
pip install rl-sales-augment
```

## Quickstart

```python
import rl_sales_augment as rsa

gen = rsa.providers.openai_chat(model="gpt-5.5")      # or your own: gen = lambda prompt: str
bot = rsa.load_agent(gen, company_ctx="Acme sells AcmeBox, an $8k on-prem appliance.")

out = bot.reply("honestly it feels expensive vs AWS")
out["chosen_move"]   # 'RAPPORT'  <- the RL decision
out["belief"]        # {'interest': .5, 'trust': .5, 'budget_fit': .2, 'objection': .8, ...}
out["reply"]         # the LLM's words, executing that move
```

`bot.reply()` is stateful: keep calling it, the agent remembers. For stateless use
(e.g. behind an API), pass the whole conversation in OpenAI message format:

```python
out = bot.chat([
    {"role": "user", "content": "what does it cost?"},
    {"role": "assistant", "content": "Depends on seats. How many do you need?"},
    {"role": "user", "content": "40 seats, but budget is tight"},
])
```

## Providers

```python
rsa.providers.openai_chat(model="gpt-5.5")                    # pip install rl-sales-augment[openai]
rsa.providers.anthropic_chat(model="claude-sonnet-5")         # [anthropic]
rsa.providers.gemini_vertex(project="my-gcp-project")         # [gemini]  (ADC, no API key)
rsa.providers.gemini_api()                                    # [gemini]  (GEMINI_API_KEY)
rsa.providers.gemma_e2b()                                     # [gemma]   local Gemma 4, CPU/MPS/CUDA
rsa.providers.openai_chat(base_url="http://...")              # any OpenAI-compatible server
```

Or bring your own: any `gen(prompt) -> str` works.

## MCP server

```bash
pip install "rl-sales-augment[mcp]"
rl-sales-augment-mcp        # stdio
```

```json
{ "mcpServers": { "rl-sales-augment": { "command": "rl-sales-augment-mcp" } } }
```

Tools: `next_move` (buyer state → RL move), `perception_prompt`, `list_moves`, `list_segments`.

## REST API

```bash
pip install "rl-sales-augment[gemini,api]"
```

A complete FastAPI server (`POST /v1/chat`, OpenAI-format messages) ships in
[examples/fastapi_server.py](https://github.com/NandhaKishorM/rl-sales-augment/blob/main/examples/fastapi_server.py).

## Why not just call GPT-5.6 / Opus 4.8 / Gemini directly?

Because what kills LLM sales conversations isn't the words, it's the **timing**. Frontier models
are trained to be helpful and agreeable, so on a skeptical buyer they answer every objection
politely, forever, and never risk asking for the deal (measured: **0/4 closes on adversarial
buyers** while handling every question beautifully). A bigger model writes better sentences; it
doesn't fix this, because next-token training never rewards a deal that closes six turns later.

The bundled policy is different in kind, not degree:

- **Trained on outcomes, not text.** PPO over millions of simulated deals with delayed, stochastic
  rewards. It has *lost* deals to premature pitching, burned reputation on spam-closing, and learned
  that discounting converts SMBs but insults enterprise. An API model has read about selling; the
  policy has sold.
- **State-dependent timing.** Prompting "be assertive, always close" makes a bot uniformly pushy.
  The skill is *when*: the policy closes at high readiness and keeps building trust below it. Same
  LLM writing the words, right moment to ask. Result: 100% vs 19-31% close in a paired A/B, 3/4 vs
  0/4 on hard buyers (simulated; harness in the repo).
- **Consistent and auditable.** Sampled LLM strategy swings run-to-run; the policy is deterministic,
  and every turn logs `chosen_move` + `belief`, so you can see *why* it did what it did.
- **Complementary and tiny.** A ~1MB MLP on CPU. Keep GPT-5.6 / Opus 4.8 / Gemini for language,
  empathy, and knowledge; add the decision layer they don't have. Retrainable on your own funnel's
  economics (the commercial offering).

Details, transcripts, and a 67-second demo:
**[github.com/NandhaKishorM/rl-sales-augment](https://github.com/NandhaKishorM/rl-sales-augment)**

## License

AGPL-3.0-or-later. Training the policy on your own market is the commercial offering:
nandakishor@convaiinnovations.com (Convai Innovations Pvt. Ltd.).
