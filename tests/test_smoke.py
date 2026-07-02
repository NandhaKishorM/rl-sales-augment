"""Offline smoke test: core install (numpy + torch) + the bundled model, mock LLM."""
import os
import rl_sales_augment as rsa


def mock_gen(prompt: str) -> str:
    if "Return ONLY JSON" in prompt:
        return '{"interest":0.6,"trust":0.5,"budget_fit":0.4,"objection":0.6,"patience":0.7}'
    return "Fair point. What would make this an easy yes for your team?"


def test_constants_and_model():
    assert len(rsa.ACTION_NAMES) == 8 and "CLOSE" in rsa.ACTION_NAMES
    assert len(rsa.SEG_NAMES) == 10
    assert rsa.MODEL_PATH.endswith("rl_sales_agent.pt")
    assert os.path.exists(rsa.MODEL_PATH)


def test_load_and_reply():
    bot = rsa.load_agent(mock_gen, company_ctx="Test Co sells widgets.")
    bot.new_conversation(segment=7)
    out = bot.reply("this seems expensive")
    assert out["chosen_move"] in rsa.ACTION_NAMES
    assert isinstance(out["reply"], str) and out["reply"].strip()
    assert {"interest", "trust", "budget_fit", "objection", "patience"} <= set(out["belief"])
    # memory persists across turns
    out2 = bot.reply("ok tell me more")
    assert out2["history_len"] > out["history_len"]


def test_perception_only():
    belief = rsa.estimate_state_via(mock_gen, [{"role": "customer", "text": "too pricey"}])
    assert 0.0 <= belief["objection"] <= 1.0


def test_chat_native_path():
    """A chat-capable generate_fn must receive a system instruction + the history as chat turns."""
    seen = {}

    def chat_gen(prompt="", *, system=None, history=None):
        if "Return ONLY JSON" in (prompt or ""):
            return '{"interest":0.6,"trust":0.5,"budget_fit":0.4,"objection":0.6,"patience":0.7}'
        seen["system"], seen["history"] = system, history
        return "Sure, let's find the right fit for you."

    bot = rsa.load_agent(chat_gen, company_ctx="Acme sells widgets.")
    bot.new_conversation(segment=7)
    bot.reply("hi there")
    bot.reply("this seems pricey")
    assert seen["system"] and "move" in seen["system"].lower()          # RL move is in the system prompt
    assert seen["history"][-1] == {"role": "user", "content": "this seems pricey"}   # current turn last
    assert any(m["role"] == "assistant" for m in seen["history"])       # prior bot turn is in the history


def test_chatgpt_template():
    """chat(messages): OpenAI-format multi-turn in, next assistant turn out; stateless per call."""
    bot = rsa.load_agent(mock_gen, company_ctx="Acme sells widgets.")
    out = bot.chat([
        {"role": "system", "content": "Discounts capped at 10%."},
        {"role": "user", "content": "what does it cost?"},
        {"role": "assistant", "content": "Depends on seats. How many do you need?"},
        {"role": "user", "content": "40 seats, but budget is tight"},
    ])
    assert out["chosen_move"] in rsa.ACTION_NAMES and out["reply"].strip()
    assert out["history_len"] == 4          # 3 prior turns rebuilt + this reply
    # must reject a conversation that doesn't end with the user
    try:
        bot.chat([{"role": "assistant", "content": "hi"}])
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_constants_and_model()
    test_load_and_reply()
    test_perception_only()
    test_chat_native_path()
    test_chatgpt_template()
    print("all smoke tests passed")
