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


def test_env_loader():
    """load_env fills os.environ from a .env file but never overrides real env vars."""
    import os, tempfile
    os.environ["RSA_TEST_EXISTING"] = "keep"
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("# comment\nRSA_TEST_NEW='from-file'\nRSA_TEST_EXISTING=clobber\nexport RSA_TEST_EXPORTED=ok\n")
        path = f.name
    added = rsa.load_env(path)
    assert os.environ["RSA_TEST_NEW"] == "from-file"
    assert os.environ["RSA_TEST_EXISTING"] == "keep"          # real env wins
    assert os.environ["RSA_TEST_EXPORTED"] == "ok"            # 'export ' prefix handled
    assert "RSA_TEST_NEW" in added and "RSA_TEST_EXISTING" not in added
    assert rsa.load_env(path) == {}                           # second load is a no-op
    os.unlink(path)


def test_slop_stripper():
    """_clean removes leading canned-validation tics but never guts a short reply."""
    from rl_sales_augment._text import _clean
    assert _clean("That's a great question! The box ships with rails and cables included.") == \
        "The box ships with rails and cables included."
    assert _clean("I completely understand, pricing feels steep until you see the TCO math.") == \
        "Pricing feels steep until you see the TCO math."
    assert _clean("That makes total sense.") == "That makes total sense."   # too short to strip
    assert _clean("That makes sense, AWS can seem pretty pricey upfront to most teams.") == \
        "AWS can seem pretty pricey upfront to most teams."
    assert _clean("That's fair. Want the real numbers?") == "That's fair. Want the real numbers?"
    assert _clean("Sure, Tuesday works for the demo.") == "Sure, Tuesday works for the demo."
    assert _clean("I don\\'t have the exact number here, we\\'ll confirm it today.") == \
        "I don't have the exact number here, we'll confirm it today."
    from rl_sales_augment.style import heuristic_style_score
    assert heuristic_style_score("You're absolutely right, great question!") < \
        heuristic_style_score("Fair point. Want me to send the real numbers?")


def test_price_grounding():
    """An invented price triggers one regeneration; a stubborn one is flagged, never silent."""
    calls = {"n": 0}
    def liar_then_honest(prompt="", **kw):
        if "Return ONLY JSON" in (prompt or ""):
            return '{"interest":0.5,"trust":0.5,"budget_fit":0.5,"objection":0.3,"patience":0.7}'
        calls["n"] += 1
        return ("The full package is $950 and standalone is $499." if calls["n"] == 1
                else "I'll confirm the exact pricing and get back to you today, what would you use it for?")
    bot = rsa.load_agent(liar_then_honest, company_ctx="Convai sells Nadhi, a desktop AI co-scientist.")
    out = bot.reply("the subscription fee is?")
    assert "ungrounded_price" not in out and "$950" not in out["reply"]

    def stubborn(prompt="", **kw):
        if "Return ONLY JSON" in (prompt or ""):
            return '{"interest":0.5,"trust":0.5,"budget_fit":0.5,"objection":0.3,"patience":0.7}'
        return "It costs $950 for the full package, a very fair deal."
    bot2 = rsa.load_agent(stubborn, company_ctx="Convai sells Nadhi.")
    out2 = bot2.reply("price?")
    assert out2["ungrounded_price"] == [950.0]     # surfaced, never silent

    def grounded(prompt="", **kw):
        if "Return ONLY JSON" in (prompt or ""):
            return '{"interest":0.5,"trust":0.5,"budget_fit":0.5,"objection":0.3,"patience":0.7}'
        return "NimbusBox is about $8,000 one-time."
    bot3 = rsa.load_agent(grounded, company_ctx="NimbusBox costs about $8k.")
    out3 = bot3.reply("cost?")
    assert "ungrounded_price" not in out3          # $8k == $8,000, no false positive


def test_script_leak_guard():
    """A stray-CJK draft triggers one regeneration; legit multilingual replies pass."""
    calls = {"n": 0}
    def glitchy_then_clean(prompt="", **kw):
        if "Return ONLY JSON" in (prompt or ""):
            return '{"interest":0.5,"trust":0.5,"budget_fit":0.5,"objection":0.3,"patience":0.7}'
        calls["n"] += 1
        return ("We are a technology公司 focused on local AI tools for research teams." if calls["n"] == 1
                else "We build local AI tools for research teams, fully on your own machine.")
    bot = rsa.load_agent(glitchy_then_clean, company_ctx="Convai sells Nadhi.")
    out = bot.reply("what do you do?")
    assert "script_mismatch" not in out and "公司" not in out["reply"]


if __name__ == "__main__":
    test_constants_and_model()
    test_load_and_reply()
    test_perception_only()
    test_chat_native_path()
    test_chatgpt_template()
    test_env_loader()
    test_slop_stripper()
    test_price_grounding()
    test_script_leak_guard()
    print("all smoke tests passed")
