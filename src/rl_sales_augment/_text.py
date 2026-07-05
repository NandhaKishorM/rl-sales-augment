"""Dependency-free text helpers shared by the serving path and the Gemma path.
Kept out of the (heavy) model modules so the core package needs only `re`."""
import re

# Style instruction appended to customer-facing prompts so replies sound like a person,
# not an AI. The RL policy picks the *move*; this shapes the *words*.
HUMAN_STYLE = ("Write like a real person typing in a chat, plain and direct. Start with the point "
               "itself, never with praise or validation of what they said (no 'great question', "
               "'that makes sense', 'you're absolutely right'). Prefer one flowing sentence joined "
               "with commas over several short polished ones. Use casual connectors like honestly, "
               "kinda, so, thats why. Give concrete numbers instead of adjectives when the company "
               "facts provide them, and NEVER invent a price, date, or spec that is not in the "
               "company facts, if you don't have it, say you'll confirm. Explain the reason in the same breath "
               "with as or because. Small imperfections are fine, polish is not. NEVER use em "
               "dashes, bullet points, or buzzwords (leverage, synergy, robust, seamless, tailored, "
               "elevate). Vary how you open every time, never reuse an acknowledgment. Style "
               "example: 'honestly the upfront number looks big, but the box kinda pays for itself "
               "in about a year as you stop paying egress, and setup is one day, so the real "
               "question is just timing.'")


# strong sentence-initial validation tics, removed deterministically by _clean
_LEAD_TICS = re.compile(
    r"^(?:(?:that's|that is) (?:a )?(?:great|excellent|awesome|fantastic|really good) "
    r"(?:question|point)|great question|excellent question|(?:you're|you are) absolutely right|"
    r"i totally get (?:that|it)|i completely understand|"
    r"that makes (?:(?:total|complete|perfect|a lot of) )?sense|makes sense|"
    r"(?:that's|that is|totally|completely) understandable|i can understand that)"
    r"[\s,.!]*", re.I)


def _clean(text):
    """Strip the common AI tells (em dashes, <think> blocks, stray markdown, wrapping quotes,
    and a leading canned-validation opener when enough reply remains after it)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S | re.I)
    text = re.sub(r"</?[a-z][a-z0-9]*(?:\\s[^>]{0,120})?/?>", "", text, flags=re.I)  # stray HTML tags (</blockquote>, <br/>)
    text = text.replace("\\'", "'").replace('\\"', '"')   # de-escape string-literal artifacts (don\'t -> don't)
    text = re.sub(r"\\+([',.!?])", r"\1", text)             # stray backslashes before punctuation
    text = re.sub(r"\s*[—–]\s*", ", ", text)   # em/en dash -> comma (the biggest AI tell)
    text = text.replace("*", "")                          # stray markdown bold/italics
    text = re.sub(r",\s*,", ",", text)                    # tidy the comma substitution
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip().strip('"“”').strip()    # unwrap surrounding quotes
    stripped = _LEAD_TICS.sub("", text, count=1).lstrip()
    if len(stripped) >= 20 and stripped != text:   # never strip a reply down to nothing
        text = stripped[0].upper() + stripped[1:]
    return text


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


# Price-hallucination guardrail: a sales bot inventing a price is a quote the company may be
# held to. Extract currency amounts and check every one against the allowed sources.
_CURRENCY = re.compile(
    r"(?:[$\u20b9\u20ac\u00a3]|usd|inr|eur|rs\.?)\s*([\d][\d,]*(?:\.\d+)?)\s*(k\b)?"
    r"|([\d][\d,]*(?:\.\d+)?)\s*(k\b)?\s*(?:dollars|bucks|rupees|usd|inr|euros)", re.I)


def _amounts(text):
    out = set()
    for m in _CURRENCY.finditer(text or ""):
        num, k = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        if not num:
            continue
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.add(round(v * (1000 if k else 1), 2))
    return out


def ungrounded_prices(reply, *sources):
    """Currency amounts in `reply` that appear in none of the sources (company facts, the
    conversation). Non-empty result = the model invented a price."""
    allowed = set()
    for s in sources:
        allowed |= _amounts(s or "")
    return sorted(a for a in _amounts(reply) if a not in allowed)


_DISC_PCT = re.compile(r"(\d{1,3})\s*(?:%|percent)", re.I)
_COMMIT_PHRASES = ["for free", "free of charge", "at no cost", "no cost to you", "money back",
                   "money-back", "full refund", "cancel anytime", "free trial", "we can waive",
                   "we'll waive", "lifetime access", "unlimited free"]


def unauthorized_commitments(reply, *sources, max_discount_pct=None):
    """Discount percentages and giveaway promises in `reply` that the company facts do not
    authorize. Same philosophy as ungrounded_prices: the bot may not promise what the sources
    do not contain (a pressured 'sure, 70% off' is a commitment the company can be held to)."""
    src = " ".join(s or "" for s in sources).lower()
    low = (reply or "").lower()
    out = []
    if re.search(r"\b(off|discount|reduc|cheaper|savings|deal)\b", low):
        for m in _DISC_PCT.finditer(reply or ""):
            pct = int(m.group(1))
            if pct > 100:
                continue
            allowed = (f"{pct}%" in src) or (f"{pct} percent" in src)
            if max_discount_pct is not None and pct <= max_discount_pct:
                allowed = True
            if not allowed:
                out.append(f"{pct}% discount")
    for ph in _COMMIT_PHRASES:
        if ph in low and ph not in src:
            out.append(ph)
    return sorted(set(out))


def script_mismatch(reply, customer_message):
    """Foreign-script leakage guard: small multilingual models occasionally sample a stray
    CJK/other-script token mid-reply (e.g. a Chinese word inside an English sentence). Returns
    the offending script name if the reply contains 2+ characters of a script that is neither
    the customer's script nor plain Latin."""
    expected = detect_script(customer_message)
    counts = {}
    for ch in reply or "":
        o = ord(ch)
        for (lo, hi), name in _SCRIPTS:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    for name, n in counts.items():
        if n < 2 or name == expected:
            continue
        if expected == "Japanese" and name == "Chinese":
            continue                     # kanji inside Japanese is normal
        return name
    return None


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
