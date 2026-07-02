"""Same agent, real LLMs. Each provider is a one-liner; install the matching extra first.
  pip install "rl-sales-augment[gemini]"     (or [openai], [anthropic], [gemma])
"""
import rl_sales_augment as rsa

COMPANY = """
Company: NimbusEdge - edge & cloud infra (Bengaluru, Austin, Singapore).
Products: NimbusBox (on-prem appliance, ~$8k), NimbusOne (edge SaaS, ~$2k/mo).
Edge: 1-day deploy, ~30% lower TCO than hyperscalers.
"""

# --- pick ONE ---
gen = rsa.providers.gemini_vertex(project="my-gcp-project")        # gcloud ADC, no API key
# gen = rsa.providers.gemini_api()                                  # reads GEMINI_API_KEY
# gen = rsa.providers.openai_chat(model="gpt-4o")                   # reads OPENAI_API_KEY
# gen = rsa.providers.anthropic_chat(model="claude-opus-4-8")       # reads ANTHROPIC_API_KEY
# gen = rsa.providers.gemma_e2b()                                    # Gemma 4 E2B, needs [gemma]
# gen = rsa.providers.local_gemma("/path/to/any-hf-causal-lm")      # any local HF model, needs [gemma]

bot = rsa.load_agent(gen, company_ctx=COMPANY, rerank_n=1)
bot.new_conversation(segment=7)

conversation = [
    "hey, just looking around at options for our datacenter",
    "we keep getting random crashes and it's killing us",
    "what would something like this actually cost",
    "ok my team is on board, what's the next step",
]
for msg in conversation:
    out = bot.reply(msg)
    print(f"\nCustomer: {msg}")
    print(f"  [{out['chosen_move']}] {out['reply']}")
