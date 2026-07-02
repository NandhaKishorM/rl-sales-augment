"""Dependency-free text helpers shared by the serving path and the Gemma path.
Kept out of the (heavy) model modules so the core package needs only `re`."""
import re

# Style instruction appended to customer-facing prompts so replies sound like a person,
# not an AI. The RL policy picks the *move*; this shapes the *words*.
HUMAN_STYLE = ("Talk like a real person on a live call, not a marketing script: use contractions, "
               "everyday words and short sentences, and be a little informal and imperfect. NEVER "
               "use em dashes, bullet points, or buzzwords (leverage, synergy, robust, seamless, "
               "tailored, elevate). Vary how you open every time, and do NOT reuse the same "
               "acknowledgment (never keep saying 'I totally get that' or 'yeah, I get that'). "
               "Don't sound scripted.")


def _clean(text):
    """Strip the common AI tells (em dashes, <think> blocks, stray markdown, wrapping quotes)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S | re.I)
    text = re.sub(r"\s*[—–]\s*", ", ", text)   # em/en dash -> comma (the biggest AI tell)
    text = text.replace("*", "")                          # stray markdown bold/italics
    text = re.sub(r",\s*,", ",", text)                    # tidy the comma substitution
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip().strip('"“”').strip()    # unwrap surrounding quotes
