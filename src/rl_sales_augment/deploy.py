"""Transfer the RL sales experience into frozen Gemma -> a real-time customer bot.

The trained RL policy IS the experience: from thousands of noisy quarters it has
learned which move actually works next, for THIS segment, in THIS deal state. In a
live chatbot/voicebot the loop is:

    customer message + deal state
              |
     RL policy  ->  best strategic move (e.g. HANDLE_OBJECTION for an enterprise
              |      security deal; DISCOUNT for a price-sensitive hardware SMB)
              |      + a hidden 'experience' vector
              v
    frozen Gemma  ->  the actual words the bot says, executing that move for this
                      segment, grounded in the injected experience latent + briefing.

So the LLM handles language and empathy; the RL policy supplies the strategy the LLM
cannot learn from priors. `bot_reply(..., mode='voice')` returns a short spoken line
for a voicebot; `mode='chat'` returns a longer written reply. `rl_augmented_sales_advice`
is the manager-level pipeline overview.
"""
from __future__ import annotations
import inspect
import torch
from .world import (SalesWorld, SalesConfig, ACTION_NAMES, SEG_NAMES, N_ACTIONS,
                         RESEARCH, RAPPORT, PITCH, HANDLE_OBJECTION, DISCOUNT, FOLLOW_UP, CLOSE, DROP)
from ._text import HUMAN_STYLE, language_instruction, ungrounded_prices, script_mismatch


def _supports_chat(fn):
    """Does this generate_fn accept the richer (system=, history=) chat interface?"""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return True
    return "system" in params and "history" in params


def _to_messages(history):
    """Agent memory [{'role':'customer'/'bot','text':..}] -> chat [{'role':'user'/'assistant','content':..}]."""
    return [{"role": "user" if t["role"] == "customer" else "assistant", "content": t["text"]}
            for t in history]


def _llm_call(gen, system, messages):
    """Call the LLM with a system instruction + conversation history. Uses the native chat interface
    (roles + turns) when the provider supports it; otherwise flattens to one prompt so a simple
    prompt->str function still works with full context."""
    if _supports_chat(gen):
        return gen("", system=system, history=messages)
    convo = "\n".join(f"{'Customer' if m['role'] == 'user' else 'Rep'}: {m['content']}" for m in messages)
    return gen(f"{system}\n\nConversation so far:\n{convo}\n\nWrite your next reply as the rep.")


# what each learned move should accomplish in the customer conversation
MOVE_INTENT = {
    RESEARCH: "ask one light qualifying question to understand their situation",
    RAPPORT: "build rapport and credibility; do NOT pitch or push yet",
    PITCH: "present the value proposition, tailored to their use case",
    HANDLE_OBJECTION: "acknowledge their concern and resolve it directly and honestly",
    DISCOUNT: "introduce pricing flexibility or an incentive to improve affordability",
    FOLLOW_UP: "keep it warm and propose a concrete, low-friction next step",
    CLOSE: "confidently ask for the commitment and lay out the steps to get started",
    DROP: "politely disengage while leaving the door open for the future",
}


@torch.no_grad()
def next_moves(agent, featurizer, world: SalesWorld):
    """Greedy move per deal + the policy's hidden 'experience' representation."""
    z = featurizer.featurize([world._obs()], [world.describe()])
    h = agent.body(z)                                   # (1, hidden)
    logits = agent.actor(h).view(world.n, N_ACTIONS)
    actions = logits.argmax(-1).cpu().numpy().tolist()
    return actions, h


def _experience_hidden(agent, featurizer, world, bridge, device):
    actions, h = next_moves(agent, featurizer, world)
    oh = torch.zeros(1, world.n * N_ACTIONS, device=h.device)
    for i, a in enumerate(actions):
        oh[0, i * N_ACTIONS + a] = 1.0
    return bridge(torch.cat([h, oh], -1).to(device)), actions


def experience_dim(hidden_dim, n_leads):
    return hidden_dim + n_leads * N_ACTIONS


def _format_history(history, last_n=12):
    turns = history[-last_n:]
    return "\n".join(f"{'Customer' if t['role']=='customer' else 'Rep'}: {t['text']}" for t in turns)


def _state_prompt(history):
    return ("Read this sales conversation and rate the BUYER'S CURRENT state as JSON: five numbers "
            "from 0.0 to 1.0 -- interest (how keen they are), trust (how much they trust the rep), "
            "budget_fit (how affordable/willing to pay it feels), objection (unresolved concerns; "
            "0=none, 1=many), patience (1=engaged, 0=about to walk). Return ONLY JSON like "
            '{"interest":0.4,"trust":0.3,"budget_fit":0.5,"objection":0.6,"patience":0.7}.\n\n'
            "CONVERSATION:\n" + _format_history(history, 12) + "\n\nJSON:")


def _parse_state(txt):
    import re, json
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    return {k: float(min(1.0, max(0.0, d[k]))) for k in
            ("interest", "trust", "budget_fit", "objection", "patience")
            if isinstance(d.get(k), (int, float))}


def estimate_state_via(generate_fn, history):
    """PERCEPTION, LLM-agnostic: read the full conversation with ANY text LLM and estimate the
    buyer's disposition for the RL policy. generate_fn(prompt)->str."""
    return _parse_state(generate_fn(_state_prompt(history)))


@torch.no_grad()
def estimate_state(gemma_lm, tok, history, device="cuda"):
    """Gemma convenience wrapper around estimate_state_via (structured/greedy for clean JSON)."""
    from .gemma import chat_generate
    return estimate_state_via(lambda p: chat_generate(gemma_lm, tok, p, 110, device, structured=True), history)


# ----------------------------------------------------- real-time customer bot
def bot_reply(agent, featurizer, gemma_lm, tok, bridge, world, deal_idx,
              customer_message, mode="chat", device="cuda", company_ctx="", temperature=None,
              style_reward=None, gemma_feat=None, n_candidates=1, avoid="", history_text=""):
    """Generate the bot's next customer-facing message for one live deal: RL-chosen strategy
    injected into Gemma, company facts + conversation history grounding the words, and (if a
    style_reward + gemma_feat are given and n_candidates>1) best-of-N reranking for authenticity.
    mode='voice' -> short spoken line."""
    exp_hidden, actions = _experience_hidden(agent, featurizer, world, bridge, device)
    move = actions[deal_idx]
    l = world.leads[deal_idx]
    seg = SEG_NAMES[l.seg]
    style = ((HUMAN_STYLE + " Just one or two short spoken sentences.") if mode == "voice"
             else (HUMAN_STYLE + " Keep it to 2-3 short sentences."))
    dont = f' You just said: "{avoid[:110]}". Open differently this time.' if avoid else ""
    header = (company_ctx + "\n\n") if company_ctx else ""
    convo = (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
    prompt = (f"{header}{convo}You are a human sales rep for a {seg} offering (deal size ~${l.value:.0f}k). "
              f"Your next move is {ACTION_NAMES[move]}: {MOVE_INTENT[move]}.\n"
              f"The customer just said: \"{customer_message}\"\n"
              f"Reply to carry out that move, using the company facts and the conversation above. "
              f"Use ONLY facts stated in the company knowledge above. If the customer ASKS for a "
              f"fact that is not covered there, especially pricing, say you will confirm and get "
              f"back, never invent a number. If the customer is just sharing information, never "
              f"offer to check or confirm anything, use what they said and respond to it directly. {style}{dont} {language_instruction(customer_message)}")
    mnt = 80 if mode == "voice" else 140
    if style_reward is not None and gemma_feat is not None and n_candidates > 1:
        from .style import best_of_n
        reply = best_of_n(gemma_lm, tok, prompt, gemma_feat, style_reward, n=n_candidates, device=device,
                          exp_hidden=exp_hidden, temperature=temperature or 1.0, max_new_tokens=mnt,
                          prev_texts=(avoid,) if avoid else ())
    else:
        from .gemma import generate_with_experience
        reply = generate_with_experience(gemma_lm, tok, prompt, exp_hidden, device=device,
                                         max_new_tokens=mnt, temperature=temperature)
    return {"segment": seg, "chosen_move": ACTION_NAMES[move], "reply": reply}


# ----------------------------------------------------- manager pipeline view
def sales_briefing(world: SalesWorld, actions):
    recs = "; ".join(f"Deal {i+1} ({SEG_NAMES[world.leads[i].seg]})={ACTION_NAMES[a]}"
                     for i, a in enumerate(actions))
    return ("EXPERIENCE BRIEFING (RL policy trained on this market; it beat hand-crafted tactics "
            "several-fold on revenue):\n"
            f"- Recommended next move per deal: {recs}.\n"
            "- Segment-specific: it does NOT over-close or pitch early, which this market punishes.")


def rl_augmented_sales_advice(agent, featurizer, gemma_lm, tok, bridge, world,
                              question="Give me the play for each deal this week.", device="cuda"):
    exp_hidden, actions = _experience_hidden(agent, featurizer, world, bridge, device)
    briefing = sales_briefing(world, actions)
    prompt = (f"You are a sales advisor. Pipeline:\n{world.describe()}\n\n{briefing}\n\n"
              f"{question}\nGive concise per-deal guidance consistent with the recommended moves.")
    from .gemma import generate_with_experience
    text = generate_with_experience(gemma_lm, tok, prompt, exp_hidden, device=device)
    return {"answer": text, "recommended": [ACTION_NAMES[a] for a in actions], "briefing": briefing}


# ==================== single-file package + on-the-fly serving ====================
def save_bundle(path, agent, manifest, style_reward=None, bridge=None):
    """Bundle the whole augmented agent into ONE file: the RL policy (trained on the world
    model, then company-fine-tuned) + the learned style reward + the latent bridge + config."""
    import torch
    torch.save({"policy": agent.state_dict(),
                "style_reward": style_reward.state_dict() if style_reward is not None else None,
                "bridge": bridge.state_dict() if bridge is not None else None,
                "manifest": manifest}, path)
    return path


class SalesBot:
    """Load ONE model file + the frozen Gemma and augment replies on the fly. Keeps a single-deal
    state that advances by the moves the policy picks (a proxy for the live conversation; in
    production you'd estimate the state from real signals via the Gemma perception encoder)."""

    def __init__(self, bundle_path, gemma_feat, device="cuda", company_ctx="", segment=None,
                 n_candidates=4, perceive=True):
        if not perceive:
            raise ValueError("perceive=False (simulated state advance) requires the full world "
                             "dynamics, which ship with the private training stack, not this package.")
        import torch
        from .policy import MultiActorCritic, ScratchFeaturizer
        from .gemma import ExperienceBridge
        b = torch.load(bundle_path, map_location=device, weights_only=False)
        m = b["manifest"]
        self.agent = MultiActorCritic(m["obs_dim"], m["policy_hidden"], m["n_leads"], m["n_actions"]).to(device).eval()
        self.agent.load_state_dict(b["policy"])
        self.feat = ScratchFeaturizer(m["obs_dim"], device=device)
        self.gemma_feat, self.gemma_lm, self.tok = gemma_feat, gemma_feat.model, gemma_feat.tok
        self.bridge = ExperienceBridge(m["exp_dim"], gemma_feat.feature_dim).to(device)
        if b.get("bridge") is not None:
            self.bridge.load_state_dict(b["bridge"])
        self.style_reward = None
        if b.get("style_reward") is not None:
            from .style import StyleReward
            self.style_reward = StyleReward(gemma_feat.feature_dim).to(device)
            self.style_reward.load_state_dict(b["style_reward"])
        self.device, self.company_ctx, self.n_candidates, self.perceive = device, company_ctx, n_candidates, perceive
        self._seg = segment
        self.new_conversation(segment)

    def new_conversation(self, segment=None):
        seg = segment if segment is not None else self._seg
        self.world = SalesWorld(SalesConfig(n_leads=1, segment_ids=(seg,) if seg is not None else None))
        self.world.reset(); self.prev = ""; self.history = []
        return self

    def reply(self, customer_message, mode="chat"):
        prior = _format_history(self.history, 8)                        # context BEFORE this message
        self.history.append({"role": "customer", "text": customer_message})
        if self.perceive:                                               # ground RL state in the real convo
            est = estimate_state(self.gemma_lm, self.tok, self.history, self.device)
            for k, v in est.items():
                setattr(self.world.leads[0], k, v)
        out = bot_reply(self.agent, self.feat, self.gemma_lm, self.tok, self.bridge, self.world, 0,
                        customer_message, mode=mode, device=self.device, company_ctx=self.company_ctx,
                        style_reward=self.style_reward, gemma_feat=self.gemma_feat,
                        n_candidates=self.n_candidates, avoid=self.prev, history_text=prior)
        self.prev = out["reply"]
        self.history.append({"role": "bot", "text": out["reply"]})
        out["estimated_state"] = {k: round(getattr(self.world.leads[0], k), 2)
                                  for k in ("interest", "trust", "budget_fit", "objection", "patience")}
        return out


# ============ LLM-agnostic augmentation with internal memory (any API) ============
class AugmentedAgent:
    """Wrap ANY text LLM (ChatGPT / Gemini / Opus 4.8 / local Gemma) with the RL policy.

    The single RL file supplies the STRATEGY; this module keeps its OWN internal memory --
    the belief state (perceived interest/trust/budget/objection/patience) + the full conversation
    history -- so it works over a STATELESS API with no access to model internals. Runs anywhere
    (the policy is tiny); all LLM calls go through generate_fn(prompt)->str.

        def gpt(prompt):
            return client.chat.completions.create(model="gpt-...",
                messages=[{"role":"user","content":prompt}]).choices[0].message.content
        bot = AugmentedAgent("rl_sales_agent.pt", gpt, company_ctx=ctx)
        print(bot.reply("Honestly it feels expensive.")["reply"])
    """

    def __init__(self, bundle_path, generate_fn, device="cpu", company_ctx="", segment=None,
                 rerank_n=1, humanize=True):
        import torch
        from .policy import MultiActorCritic, ScratchFeaturizer
        b = torch.load(bundle_path, map_location=device, weights_only=False); m = b["manifest"]
        self.agent = MultiActorCritic(m["obs_dim"], m["policy_hidden"], m["n_leads"], m["n_actions"]).to(device).eval()
        self.agent.load_state_dict(b["policy"])
        self.feat = ScratchFeaturizer(m["obs_dim"], device=device)
        self.gen, self.company_ctx, self._seg = generate_fn, company_ctx, segment
        self.rerank_n, self.humanize, self.device = rerank_n, humanize, device
        self.new_conversation(segment)

    def new_conversation(self, segment=None):
        seg = segment if segment is not None else self._seg
        self.world = SalesWorld(SalesConfig(n_leads=1, segment_ids=(seg,) if seg is not None else None))
        self.world.reset(); self.history = []; self.prev = ""       # <- the internal memory
        return self

    @torch.no_grad()
    def _rl_move(self):
        z = self.feat.featurize([self.world._obs()], [self.world.describe()])
        return int(self.agent.actor(self.agent.body(z)).view(N_ACTIONS).argmax().item())

    def _finish(self, text):
        if not self.humanize:
            return text.strip()
        from ._text import _clean
        return _clean(text)

    def reply(self, customer_message, mode="chat"):
        self.history.append({"role": "customer", "text": customer_message})
        for k, v in estimate_state_via(self.gen, self.history).items():   # PERCEPTION -> belief memory
            setattr(self.world.leads[0], k, v)
        move = self._rl_move()                                            # STRATEGY: the RL-chosen move
        l = self.world.leads[0]; seg = SEG_NAMES[l.seg]
        style = HUMAN_STYLE + (" One or two short spoken sentences." if mode == "voice"
                               else " Keep it to 2-3 short sentences.")
        dont = f' You just said: "{self.prev[:110]}". Open differently.' if self.prev else ""
        head = (self.company_ctx + "\n\n") if self.company_ctx else ""
        # the system instruction carries company facts + persona + the RL move; the conversation
        # history is passed as real chat turns (see _llm_call), not flattened into this string.
        system = (f"{head}You are a human sales rep for a {seg} offering (~${l.value:.0f}k). "
                  f"Your next move is {ACTION_NAMES[move]}: {MOVE_INTENT[move]}. "
                  f"Reply to carry out that move, using the company facts and the conversation. "
                  f"Use ONLY facts stated in the company knowledge above. If the customer ASKS for a "
                  f"fact that is not covered there, especially pricing, say you will confirm and get "
                  f"back, never invent a number. If the customer is just sharing information, never "
                  f"offer to check or confirm anything, use what they said and respond to it directly. {style}{dont} {language_instruction(customer_message)}")
        messages = _to_messages(self.history)                            # MEMORY: full history as chat turns
        if self.rerank_n > 1:
            from .style import heuristic_style_score
            cands = [self._finish(_llm_call(self.gen, system, messages)) for _ in range(self.rerank_n)]
            reply = max(cands, key=lambda c: heuristic_style_score(c, (self.prev,)))
        else:
            reply = self._finish(_llm_call(self.gen, system, messages))
        # grounding guardrail: never let an invented price out the door
        sources = [self.company_ctx] + [t["text"] for t in self.history]
        invented = ungrounded_prices(reply, *sources)
        if invented:
            warn = (" IMPORTANT: your draft invented a price that is NOT in the company facts. "
                    "State no prices or amounts at all, say you will confirm exact pricing and "
                    "get back, then continue the move.")
            reply = self._finish(_llm_call(self.gen, system + warn, messages))
            invented = ungrounded_prices(reply, *sources)
        bad_script = script_mismatch(reply, customer_message)
        if bad_script:
            warn2 = (f" IMPORTANT: your draft mixed {bad_script} characters into the reply. "
                     f"Rewrite the ENTIRE reply cleanly in the customer's language only.")
            reply = self._finish(_llm_call(self.gen, system + warn2, messages))
            bad_script = script_mismatch(reply, customer_message)
        self.prev = reply; self.history.append({"role": "bot", "text": reply})
        out = {"chosen_move": ACTION_NAMES[move], "reply": reply,
               "belief": {k: round(getattr(l, k), 2)
                          for k in ("interest", "trust", "budget_fit", "objection", "patience")},
               "history_len": len(self.history)}
        if invented:
            out["ungrounded_price"] = invented     # integrators can block/route these
        if bad_script:
            out["script_mismatch"] = bad_script    # surfaced, never silent
        return out

    def chat(self, messages, mode="chat"):
        """ChatGPT-template entry point: pass the FULL conversation in OpenAI message format and
        get the next assistant turn. Stateless per call (perfect behind a REST API) -- the agent
        rebuilds its internal memory (belief + history) from `messages` every time.

            messages = [
                {"role": "system", "content": "optional extra company facts for this call"},
                {"role": "user", "content": "hey, what does this cost?"},
                {"role": "assistant", "content": "Depends on volume. What size team?"},
                {"role": "user", "content": "about 40 seats, and honestly budget is tight"},
            ]
            out = bot.chat(messages)   # -> {"chosen_move", "reply", "belief", "history_len"}

        The last message must be from the user. A leading system message (optional) is treated as
        additional company context for this call only.
        """
        msgs = list(messages or [])
        extra_ctx = ""
        if msgs and msgs[0].get("role") == "system":
            extra_ctx = str(msgs[0].get("content", "")).strip()
            msgs = msgs[1:]
        if not msgs or msgs[-1].get("role") != "user":
            raise ValueError("messages must end with a 'user' message")
        # rebuild internal memory from the prior turns
        self.new_conversation()
        for m in msgs[:-1]:
            role = "customer" if m.get("role") == "user" else "bot"
            self.history.append({"role": role, "text": str(m.get("content", ""))})
            if role == "bot":
                self.prev = str(m.get("content", ""))
        if extra_ctx:
            base_ctx = self.company_ctx
            self.company_ctx = (base_ctx + "\n" + extra_ctx).strip()
            try:
                return self.reply(str(msgs[-1]["content"]), mode=mode)
            finally:
                self.company_ctx = base_ctx
        return self.reply(str(msgs[-1]["content"]), mode=mode)
