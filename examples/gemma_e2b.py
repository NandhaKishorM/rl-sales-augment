"""Google Gemma 4 E2B, two ways.  Install:  pip install "rl-sales-augment[gemma]"
Pass a local path to either call to reuse a downloaded copy instead of pulling from HF.
"""
import rl_sales_augment as rsa

COMPANY = "Acme Edge sells the AcmeBox on-prem datacenter appliance (~$8k), 1-day deploy."
GEMMA = "google/gemma-4-E2B-it"     # or a local path e.g. "/path/to/gemma-4-E2B-it"

# --- Route 1: Gemma as the words-writer for the portable agent (prompt-level augmentation) ---
print("=== Route 1: load_agent + gemma_e2b provider ===")
gen = rsa.providers.gemma_e2b(GEMMA)              # runs on MPS / CUDA / CPU, bf16 off-CPU
bot = rsa.load_agent(gen, company_ctx=COMPANY)
bot.new_conversation(segment=7)
out = bot.reply("honestly it feels expensive compared to just using cloud")
print(f"[{out['chosen_move']}] {out['reply']}")
print("belief:", out["belief"])

# --- Route 2: Gemma-native SalesBot (experience bridge + trained style reranker) ---
# print("\n=== Route 2: load_gemma_bot (Gemma-native) ===")
# sb = rsa.load_gemma_bot(GEMMA, company_ctx=COMPANY, n_candidates=1)
# sb.new_conversation(segment=7)
# o2 = sb.reply("we keep getting random crashes and it is hurting us")
# print(f"[{o2['chosen_move']}] {o2['reply']}")
# print("estimated_state:", o2["estimated_state"])
