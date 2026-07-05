"""Fully local, no API key: Gemma 4 E4B (the bridge-aligned flagship) writes the words.
E2B for E2B, E4B for E4B: this is the E4B twin of gemma_e2b.py."""
import rl_sales_augment as rsa

COMPANY = "Acme Edge sells the AcmeBox on-prem datacenter appliance (~$8k), 1-day deploy."

gen = rsa.providers.gemma_e4b()                    # google/gemma-4-E4B-it, ~16 GB one-time
bot = rsa.load_agent(gen, company_ctx=COMPANY)
out = bot.reply("honestly it feels expensive vs AWS")
print(out["chosen_move"], "->", out["reply"])
