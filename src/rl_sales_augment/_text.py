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


# Unicode-script detector: name the customer's script so even small LLMs mirror the language.
_SCRIPTS = [
    ((0x0D00, 0x0D7F), "Malayalam"), ((0x0B80, 0x0BFF), "Tamil"), ((0x0900, 0x097F), "Devanagari (Hindi/Marathi)"),
    ((0x0C00, 0x0C7F), "Telugu"), ((0x0C80, 0x0CFF), "Kannada"), ((0x0980, 0x09FF), "Bengali"),
    ((0x0A80, 0x0AFF), "Gujarati"), ((0x0A00, 0x0A7F), "Gurmukhi (Punjabi)"), ((0x0B00, 0x0B7F), "Odia"),
    ((0x3040, 0x30FF), "Japanese"), ((0xAC00, 0xD7AF), "Korean"), ((0x0600, 0x06FF), "Arabic"),
    ((0x0400, 0x04FF), "Cyrillic (Russian)"), ((0x0E00, 0x0E7F), "Thai"), ((0x0370, 0x03FF), "Greek"),
    ((0x4E00, 0x9FFF), "Chinese"),
]

def detect_script(text):
    """Best-effort script/language name from Unicode ranges; None for Latin-script text."""
    counts = {}
    for ch in text:
        o = ord(ch)
        for (lo, hi), name in _SCRIPTS:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    best = max(counts, key=counts.get)
    return best if counts[best] >= 3 else None


def language_instruction(customer_message):
    """A reply-language directive sized for small models: explicit script name when detectable."""
    script = detect_script(customer_message)
    if script:
        return (f"IMPORTANT: the customer is writing in {script} script. Write your ENTIRE reply in "
                f"the customer's language, using {script} script. Do not reply in English.")
    return "IMPORTANT: write your ENTIRE reply in the same language as the customer's most recent message."
