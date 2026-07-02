"""`generate_fn` factories: turn any LLM into the callable the agent needs.

Each factory returns a `gen(prompt="", *, system=None, history=None) -> str` closure. The agent
calls it two ways:
  * perception (one-shot):  gen(prompt)                      -> a single user message
  * reply (multi-turn):     gen(system=..., history=[...])   -> a native chat with roles

`history` is a list of `{"role": "user"|"assistant", "content": str}` (the conversation so far),
so every provider sends the real turn structure as its chat template -- not history flattened into
one string. A prompt containing "Return ONLY JSON" (the perception step) is decoded greedily.

Every factory takes a `model` argument. Defaults reflect mid-2026 lineups; pass any model id you
have access to. All SDK imports are lazy -- importing this module never needs the provider libraries.
Install what you use:  pip install "rl-sales-augment[gemini]"  (or [openai], [anthropic], [gemma]).

Bring your own model: pass any `gen(prompt)->str` (or the richer signature) to `load_agent`, or use
`openai_chat(base_url=...)` for any OpenAI-compatible endpoint (vLLM, Together, Groq, OpenRouter, ...).
"""
from __future__ import annotations

_JSON_HINT = "Return ONLY JSON"   # emitted by the perception prompt; triggers greedy decoding


def _openai_messages(prompt, system, history):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if history:
        msgs.extend({"role": m["role"], "content": m["content"]} for m in history)
    if prompt:
        msgs.append({"role": "user", "content": prompt})
    return msgs


# ------------------------------------------------------------------ Gemini (Vertex or API key)
def _gemini_gen(client, model, base_tokens):
    from google.genai import types
    import time

    def _contents(prompt, history):
        c = []
        if history:
            for m in history:
                c.append({"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]})
        if prompt:
            c.append({"role": "user", "parts": [{"text": prompt}]})
        return c or (prompt or " ")

    def gen(prompt: str = "", *, system=None, history=None) -> str:
        structured = _JSON_HINT in (prompt or "")
        contents = _contents(prompt, history)
        last = ""
        for attempt in range(5):
            cfg = types.GenerateContentConfig(
                temperature=0.0 if structured else 0.9, top_p=0.95,
                max_output_tokens=base_tokens * (attempt + 1),
                thinking_config=types.ThinkingConfig(thinking_budget=0),   # off -> fast, avoids empty
                system_instruction=system or None)
            try:
                r = client.models.generate_content(model=model, contents=contents, config=cfg)
                last = (r.text or "").strip()
                if last:
                    return last
            except Exception:
                time.sleep(1.2 * (attempt + 1))
        return last
    return gen


def gemini_vertex(project, location="global", model="gemini-3.5-flash", base_tokens=256):
    """Gemini via Vertex AI using Application Default Credentials (gcloud auth) -- no API key.
    Latest ids (mid-2026): gemini-3.5-flash (default), gemini-3.5-pro, gemini-3.1-pro. Needs [gemini]."""
    from google import genai
    return _gemini_gen(genai.Client(vertexai=True, project=project, location=location), model, base_tokens)


def gemini_api(api_key=None, model="gemini-3.5-flash", base_tokens=256):
    """Gemini via the Gemini Developer API (AI Studio key; reads GEMINI_API_KEY if None). Needs [gemini]."""
    import os
    from google import genai
    return _gemini_gen(genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"]), model, base_tokens)


# ------------------------------------------------------------------ OpenAI / OpenAI-compatible
def openai_chat(model="gpt-5.5", api_key=None, base_url=None, base_tokens=400):
    """OpenAI chat models (reads OPENAI_API_KEY if api_key is None). Latest ids (mid-2026):
    gpt-5.5 (default), gpt-5.4, gpt-5.4-mini, gpt-5.6-* (preview).
    Set `base_url` to point at ANY OpenAI-compatible API (vLLM, Together, Groq, OpenRouter, local).
    Needs [openai]."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    def gen(prompt: str = "", *, system=None, history=None) -> str:
        structured = _JSON_HINT in (prompt or "")
        r = client.chat.completions.create(
            model=model, messages=_openai_messages(prompt, system, history),
            temperature=0.0 if structured else 0.9, max_tokens=base_tokens)
        return (r.choices[0].message.content or "").strip()
    return gen


# ------------------------------------------------------------------ Anthropic
def anthropic_chat(model="claude-sonnet-5", api_key=None, base_tokens=512):
    """Anthropic Claude models (reads ANTHROPIC_API_KEY if None). Latest ids (mid-2026):
    claude-sonnet-5 (default), claude-opus-4-8, claude-haiku-4-5. Needs [anthropic]."""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    def gen(prompt: str = "", *, system=None, history=None) -> str:
        structured = _JSON_HINT in (prompt or "")
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
        if prompt:
            msgs.append({"role": "user", "content": prompt})
        if not msgs:                                   # Anthropic needs >=1 message
            msgs = [{"role": "user", "content": system or " "}]
        kw = {"system": system} if system else {}
        r = client.messages.create(model=model, max_tokens=base_tokens,
                                   temperature=0.0 if structured else 0.9, messages=msgs, **kw)
        return "".join(b.text for b in r.content if getattr(b, "type", None) == "text").strip()
    return gen


# ------------------------------------------------------------------ local Gemma / any HF causal LM
def local_gemma(model_path, device=None, base_tokens=256):
    """Local Gemma (or any HF causal LM) via transformers. Use AutoTokenizer (NOT AutoProcessor:
    Gemma 4's processor pulls torchvision). Runs on MPS / CUDA / CPU. Needs [gemma]."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if device is None:
        device = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()

    @torch.no_grad()
    def gen(prompt: str = "", *, system=None, history=None) -> str:
        structured = _JSON_HINT in (prompt or "")
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
        if prompt:
            msgs.append({"role": "user", "content": prompt})
        if system:                       # Gemma has no system role -> fold into the first user turn
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = system + "\n\n" + m["content"]
                    break
            else:
                msgs.insert(0, {"role": "user", "content": system})
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        gk = (dict(do_sample=False) if structured else
              dict(do_sample=True, temperature=0.9, top_p=0.92, repetition_penalty=1.2, no_repeat_ngram_size=3))
        out = model.generate(**inp, max_new_tokens=base_tokens, pad_token_id=tok.eos_token_id, **gk)
        return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen


def gemma_e2b(model="google/gemma-4-E2B-it", device=None, base_tokens=256):
    """Google Gemma 4 E2B as a generate_fn (the model this policy was trained alongside).
    `model` may be a local path or a HF repo id (auto-downloaded, not gated). Needs [gemma].

        gen = gemma_e2b()                                  # download + run google/gemma-4-E2B-it
        gen = gemma_e2b("/path/to/local/gemma-4-E2B-it")   # use a local copy
    """
    return local_gemma(model, device=device, base_tokens=base_tokens)
