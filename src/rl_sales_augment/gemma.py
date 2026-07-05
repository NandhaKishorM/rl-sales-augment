"""The common latent space: a two-way bridge between the frozen Gemma 4 E2B and
the RL Experience Model.

Direction 1 -- PERCEPTION  (LLM -> RL):
    The frozen LLM reads the world's natural-language `describe()` and its last
    hidden state becomes the RL feature. The Experience Model encodes THAT into
    its latent z. So the agent's world model is built on top of the LLM's
    understanding of the situation.

Direction 2 -- EXPERIENCE  (RL -> LLM):
    The trained Experience Model produces, for a situation, an "experience
    vector" (latent z + value + look-ahead summary). We project it into Gemma's
    hidden_size and splice it into the prompt as SOFT TOKENS via `inputs_embeds`.
    Now when the frozen LLM answers, it is conditioned on a module that has
    actually lived the consequences -- the basis for *trusting* the LLM's reply.

Gemma 4 E2B facts used (verified from the model card / transformers docs):
  * 35 decoder layers, hidden_size read at runtime from `config.hidden_size`.
  * Per-Layer Embeddings: for positions without token IDs (our soft tokens) the
    model uses the context-aware projection path, so injecting embeddings is the
    supported route (same path multimodal soft tokens use).
  * `inputs_embeds` + `generate` is supported -> soft-prompt injection works.

GPU + transformers + HF access to google/gemma-4-E4B-it required. Frozen: no base
weight ever gets a gradient; only the small projections train.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# Direction 1: perception  (frozen Gemma -> RL feature)
# ======================================================================
def text_hidden_size(config):
    """Gemma 4 is multimodal, so Gemma4Config has no top-level `hidden_size` --
    it lives on the text sub-config. Resolve it robustly across transformers
    versions. (hasattr returns False when the attribute access raises.)"""
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "get_text_config"):
        tc = config.get_text_config()
        if tc is not None and hasattr(tc, "hidden_size"):
            return tc.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise AttributeError("Could not find hidden_size on the model config.")


class GemmaFeaturizer:
    """Frozen Gemma 4 as a perception encoder. Loads the causal LM (so the SAME
    frozen model is reused for generation in deploy) and reads its last-layer
    hidden states via output_hidden_states -- robust for the multimodal model."""
    input_mode = "text"

    def __init__(self, model_id="google/gemma-4-E4B-it", device="cuda",
                 dtype=torch.bfloat16, max_tokens=160, cache=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)                      # FROZEN
        self.feature_dim = text_hidden_size(self.model.config)
        self._cache = {} if cache else None

    @torch.no_grad()
    def featurize(self, obs, texts: List[str]) -> torch.Tensor:
        if self._cache is not None:
            miss = [t for t in texts if t not in self._cache]
            if miss:
                f = self._encode(miss)
                for t, v in zip(miss, f):
                    self._cache[t] = v.to("cpu")
            return torch.stack([self._cache[t] for t in texts]).to(self.device).float()
        return self._encode(texts).float()

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_tokens).to(self.device)
        out = self.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]                               # (B,T,H) last layer
        last = enc["attention_mask"].sum(1) - 1
        idx = last.clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)                      # last-token pool


# ======================================================================
# Direction 2: experience  (RL latent -> frozen Gemma soft tokens)
# ======================================================================
class ExperienceBridge(nn.Module):
    """Projects the Experience Model's situation vector into ONE vector in Gemma's
    hidden space, added into the residual stream at an early decoder layer (see
    generate_with_experience). Only this projection trains; Gemma stays frozen.

    Why a residual-stream hook instead of `inputs_embeds` soft tokens: Gemma 4 uses
    Per-Layer Embeddings that require token IDs. Passing custom `inputs_embeds`
    without input_ids makes the model try to RECOVER ids by comparing against the
    whole ~262k-row embedding table -> a ~100 GB tensor (OOM). Injecting past the
    embedding/PLE stage sidesteps that entirely while still sharing a latent space."""

    def __init__(self, experience_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(experience_dim, 2 * hidden_size), nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size))
        # zero-init the output so an UNTRAINED bridge injects nothing (no corruption
        # of generation). align_bridge() grows it; until then the text briefing grounds.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, experience_vec: torch.Tensor) -> torch.Tensor:
        return self.proj(experience_vec)                  # (B, hidden_size)


def find_decoder_layers(model):
    """Locate the text decoder's ModuleList of blocks, robustly across the
    multimodal wrapper (prefer the longest stack whose blocks look like decoder
    layers). Returns None if not found (caller then skips latent injection)."""
    import torch.nn as nn
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            first = mod[0]
            if any(hasattr(first, a) for a in ("self_attn", "mlp", "feed_forward")):
                if best is None or len(mod) > len(best):
                    best = mod
    return best


@torch.no_grad()
def build_experience_vector(em, world, look_ahead_fn, horizon=8):
    """Assemble the situation's experience vector the LLM will be conditioned on:
    [ latent z | critic value | best/worst imagined returns | best-action one-hot ].
    Everything here comes from a model that has been verified against the world."""
    from world import N_ACTIONS
    dev = next(em.parameters()).device
    s = torch.as_tensor(world._obs(), device=dev).float().unsqueeze(0)
    z = em.encode(s)
    value = em.critic(z)
    scored = look_ahead_fn(em, world, horizon=horizon)
    best_ja, best_ret = scored[0]
    worst_ret = scored[-1][1]
    best_onehot = torch.zeros(1, world.n * N_ACTIONS, device=dev)
    for i, a in enumerate(best_ja):
        best_onehot[0, i * N_ACTIONS + a] = 1.0
    extras = torch.tensor([[value.item(), best_ret, worst_ret]], device=dev)
    vec = torch.cat([z, extras, best_onehot], dim=-1)
    return vec, best_ja, best_ret, worst_ret, scored


import re

# light sampling -> natural variation instead of one robotic template (structured JSON
# overrides this to greedy). repetition guards stay to avoid loops.
_GEN_KW = dict(do_sample=True, temperature=0.9, top_p=0.92, repetition_penalty=1.2, no_repeat_ngram_size=3)

def _chat_ids(tokenizer, prompt, device):
    """Apply the instruction-tuned chat template (this is what was missing — a raw
    prompt makes Gemma-it just continue/echo the text). Thinking disabled. Gemma 4 is
    multimodal, so apply_chat_template can return a BatchEncoding (dict-like) rather
    than a bare tensor -> pull out input_ids robustly."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt", enable_thinking=False)
    except TypeError:
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(enc, "input_ids"):          # BatchEncoding
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    return enc.to(device)                  # (1, T) input_ids tensor


def _clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S | re.I)
    text = re.sub(r"</?[a-z][a-z0-9]*(?:\\s[^>]{0,120})?/?>", "", text, flags=re.I)  # stray HTML tags (</blockquote>, <br/>)
    text = re.sub(r"\s*[—–]\s*", ", ", text)   # em/en dash -> comma (the biggest AI tell)
    text = text.replace("*", "")                          # stray markdown bold/italics
    text = re.sub(r",\s*,", ",", text)                    # tidy the comma substitution
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip().strip('"“”').strip()    # unwrap surrounding quotes


@torch.no_grad()
def chat_generate(gemma_lm, tokenizer, prompt: str, max_new_tokens=200, device="cuda",
                  structured=False, temperature=None) -> str:
    """Chat-templated generation (customer sim, extraction, the pure-LLM policy).
    structured=True (JSON/format output) uses PLAIN greedy -- the repetition guards
    used for prose break JSON (they block legitimately repeated 3-grams like {"name":).
    temperature: dial the naturalness/imperfection (higher = looser)."""
    ids = _chat_ids(tokenizer, prompt, device)
    attn = torch.ones_like(ids)
    gk = dict(do_sample=False) if structured else dict(_GEN_KW)
    if temperature is not None and not structured:
        gk["temperature"] = temperature
    gen = gemma_lm.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id, **gk)
    return _clean(tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def generate_with_experience(gemma_lm, tokenizer, prompt: str, exp_hidden=None,
                             inject_layer=3, max_new_tokens=220, device="cuda", temperature=None) -> str:
    """Chat-templated generation (PLE-safe: normal input_ids). If an experience vector
    is given, add it into the residual stream at an early decoder layer via a forward
    hook -- injected PAST the embedding/PLE stage. A zero-init bridge injects nothing."""
    ids = _chat_ids(tokenizer, prompt, device)
    attn = torch.ones_like(ids)
    handle = None
    if exp_hidden is not None:
        layers = find_decoder_layers(gemma_lm)
        if layers is not None:
            li = min(inject_layer, len(layers) - 1)
            eh = exp_hidden.to(device)

            def hook(module, inp, out):
                if isinstance(out, tuple):
                    return (out[0] + eh[:, None, :].to(out[0].dtype),) + tuple(out[1:])
                return out + eh[:, None, :].to(out.dtype)

            handle = layers[li].register_forward_hook(hook)
    gk = dict(_GEN_KW)
    if temperature is not None:
        gk["temperature"] = temperature
    try:
        gen = gemma_lm.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id, **gk)
    finally:
        if handle is not None:
            handle.remove()
    return _clean(tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))


def experience_briefing(world, scored, best_ja, best_ret, worst_ret) -> str:
    """A verifiable, human-readable summary of the world model's look-ahead. This
    is the experience expressed as TEXT (guaranteed grounding), to accompany the
    soft-token latent channel."""
    from world import ACTION_NAMES
    best = ", ".join(f"{world.names[i]}={ACTION_NAMES[best_ja[i]]}" for i in range(world.n))
    lines = [
        "EXPERIENCE BRIEFING (from a world model verified against the real environment):",
        f"- Imagined {len(scored)} possible joint decisions over an 8-step horizon.",
        f"- Best path: {best}  (predicted return {best_ret:.2f}).",
        f"- Worst path predicted return {worst_ret:.2f}; the choice matters by "
        f"{best_ret - worst_ret:.2f}.",
        "- This recommendation is grounded in lived consequences, not text priors.",
    ]
    return "\n".join(lines)


def rl_augmented_response(em, gemma_lm, tokenizer, bridge, world, look_ahead_fn,
                          user_question: str, device="cuda") -> dict:
    """Full RL Augmentation at inference: world-model look-ahead -> experience
    (latent soft tokens + text briefing) -> frozen Gemma writes the grounded,
    trustworthy answer."""
    vec, best_ja, best_ret, worst_ret, scored = build_experience_vector(em, world, look_ahead_fn)
    exp_hidden = bridge(vec.to(device))                  # (1, hidden_size)
    briefing = experience_briefing(world, scored, best_ja, best_ret, worst_ret)
    prompt = (f"You are the advisor to a civilization. Situation:\n{world.describe()}\n\n"
              f"{briefing}\n\nQuestion: {user_question}\n"
              f"Answer in 3-4 sentences, citing the consequences the world model foresaw.")
    text = generate_with_experience(gemma_lm, tokenizer, prompt, exp_hidden, device=device)
    return {"answer": text, "best_action": best_ja, "best_return": best_ret,
            "briefing": briefing}
