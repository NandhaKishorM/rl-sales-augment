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


# Romanized-Indic detection ("Manglish" = Malayalam typed in English letters, e.g. "ntha
# visesham, sugano"; likewise Hinglish, Tanglish). Script detection is blind to these, so we
# match high-frequency romanized marker words instead.
_ROMANIZED = {
    "Manglish (Malayalam typed in English letters)": {
        "aanu", "anu", "aano", "alle", "ille", "illa", "undu", "undo", "ntha", "entha", "enthu",
        "sugano", "sugam", "sheri", "pakshe", "vila", "kaashu", "kasu", "njan", "njangal",
        "nammalk", "namuk", "okke", "onnu", "ippo", "appo", "pinne", "athe", "athu", "ithu",
        "parayamo", "parayu", "kaanikko", "kaanikkamo", "venam", "veno", "vendam", "mathi",
        "pore", "kure", "valare", "kooduthal", "kurakkamo", "kurach", "kollam", "kidu",
        "adipoli", "chetta", "chechi", "mwone", "mone", "aayi", "ketto", "visesham", "karyam",
        "aazhcha", "mathiyayirunnille", "kanditt", "kandittu", "nte", "ente", "avide", "ivide",
    },
    "Hinglish (Hindi typed in English letters)": {
        "hai", "nahi", "nahin", "kya", "accha", "acha", "thoda", "zyada", "jyada", "chahiye",
        "lekin", "magar", "bahut", "kitna", "hum", "hamari", "aap", "kaise", "kyun", "kyu",
        "toh", "abhi", "wala", "karo", "karna", "sakte", "sakta", "agle", "hafte", "paisa",
        "mehnga", "mehenga", "theek", "thik", "yaar", "bhai", "matlab", "sahi", "badhiya",
        "mast", "kaafi", "chakkar", "batao", "dikha", "chalo", "gaya", "jata",
    },
    "Tanglish (Tamil typed in English letters)": {
        "irukku", "iruku", "enna", "ennada", "romba", "seri", "panna", "pannunga", "pannalam",
        "venum", "vendam", "epdi", "adhu", "idhu", "indha", "aana", "mudiyuma", "mudiyum",
        "konjam", "engaloda", "ungaloda", "vilai", "kammi", "semma", "machan", "vera", "sollu",
        "paaru", "paakalam", "paatha", "podhum", "kaatu", "kaatta", "bayama", "adhigam", "edhukku",
    },
}

_ROM_EXAMPLE = {
    "Manglish (Malayalam typed in English letters)":
        "customer: 'vila valare kooduthal aanu' -> you: 'Athe alle bro, vila kandal angane thonnum. "
        "Pakshe total cost nokkiyal ithu laabham aanu, njan kanichu tharaam.'",
    "Hinglish (Hindi typed in English letters)":
        "customer: 'price bahut zyada hai yaar' -> you: 'Sahi keh rahe ho bhai, pehle zyada lagta hai. "
        "Lekin total cost dekho toh ye kaafi sasta padta hai, main dikhata hoon.'",
    "Tanglish (Tamil typed in English letters)":
        "customer: 'vilai romba adhigam da' -> you: 'Seri da, vilai paatha adhigama thaan therium. "
        "Aana total cost paatha idhu romba save aagum, naan kaatren.'",
}

def detect_romanized(text):
    """Best-effort detection of Indian languages typed in Roman letters. Returns the label of
    the best-matching language when >=2 marker words hit, else None."""
    words = set(w.strip(".,!?\"'()") for w in text.lower().split())
    scores = {label: len(words & markers) for label, markers in _ROMANIZED.items()}
    label = max(scores, key=scores.get)
    return label if scores[label] >= 2 else None


def language_instruction(customer_message):
    """A reply-language directive sized for small models: explicit script name when detectable."""
    script = detect_script(customer_message)
    if script:
        return (f"IMPORTANT: the customer is writing in {script} script. Write your ENTIRE reply in "
                f"the customer's language, using {script} script. Do not reply in English.")
    rom = detect_romanized(customer_message)
    if rom:
        lang = rom.split(" ")[1].strip("(")
        return (f"IMPORTANT: the customer is typing {rom}. Reply the SAME way: casual {lang} words "
                f"written in English letters, mixing simple English tech words naturally. Do NOT "
                f"reply in plain English and do NOT use {lang} script. Style example, {_ROM_EXAMPLE[rom]}")
    return "IMPORTANT: write your ENTIRE reply in the same language as the customer's most recent message."
