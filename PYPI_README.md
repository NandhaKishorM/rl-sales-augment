# rl-sales-augment

A trained RL policy that picks the next sales move (rapport / pitch / objection / discount / close);
your LLM writes the words. The trained model ships in the package — CPU, no finetuning, works with
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

`bot.reply()` is stateful — keep calling it, the agent remembers. For stateless use
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

## Why

Pure LLMs answer objections politely forever and never ask for the sale. In a paired A/B
(simulated buyers, harness in the repo) the same LLM closed 100% with the policy vs 19–31% without;
on adversarial buyers 3/4 vs 0/4. Details, transcripts, and a 67-second demo:
**[github.com/NandhaKishorM/rl-sales-augment](https://github.com/NandhaKishorM/rl-sales-augment)**

## License

AGPL-3.0-or-later. Training the policy on your own market is the commercial offering —
nandakishor@convaiinnovations.com (Convai Innovations Pvt. Ltd.).
