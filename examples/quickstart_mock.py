"""Runs anywhere with just the core install (numpy + torch) -- no LLM, no network.
A mock generate_fn stands in for a real LLM so you can see the agent's loop:
perception -> RL move -> reply, with internal memory."""
import rl_sales_augment as rsa


def mock_gen(prompt: str) -> str:
    # The perception step asks for JSON; everything else is a normal reply.
    if "Return ONLY JSON" in prompt:
        return '{"interest":0.6,"trust":0.5,"budget_fit":0.4,"objection":0.6,"patience":0.7}'
    return "Totally fair to weigh that. What would make this an easy yes for your team?"


bot = rsa.load_agent(mock_gen, company_ctx="Acme Edge sells the AcmeBox on-prem appliance (~$8k).")
bot.new_conversation(segment=7)   # Hardware / datacenter

for msg in ["just looking, not sure we need this",
            "honestly it feels expensive versus cloud",
            "ok the reliability angle is interesting"]:
    out = bot.reply(msg)
    print(f"\nCustomer: {msg}")
    print(f"  RL move : {out['chosen_move']}")
    print(f"  belief  : {out['belief']}")
    print(f"  reply   : {out['reply']}")

print(f"\nbundled model: {rsa.MODEL_PATH}")
