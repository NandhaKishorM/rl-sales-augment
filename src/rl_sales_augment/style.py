"""Learned authenticity at inference: a style-reward head + best-of-N reranking.

The bundled model file ships a trained `StyleReward` head (P(sounds human) from the
frozen-Gemma embedding of a reply). At inference we sample N candidate replies and
keep the one the reward likes best, with a few cheap heuristic guardrails. This is a
separate objective from the sales policy: the RL learned *which move*, this learned
*how it should sound*.

The dataset construction and training for this head are part of the private
training stack (Convai Innovations Pvt. Ltd.) and are not distributed.
"""
from __future__ import annotations
import re
import torch
import torch.nn as nn

AI_TELLS = ["—", "–", "leverage", "synergy", "robust", "seamless", "tailored", "elevate",
            "utilize", "furthermore", "moreover", "cutting-edge", "state-of-the-art", "best-in-class",
            "rest assured", "as an ai", "i completely understand", "unparalleled", "holistically",
            "frictionless", "future-proof", "empower", "transformative", "end-to-end",
            # sycophancy / fake-empathy tells: unearned praise and canned validation
            "great question", "excellent question", "awesome question", "great point",
            "you're absolutely right", "you are absolutely right", "absolutely right",
            "that makes total sense", "makes complete sense", "that makes sense", "i totally get",
            "totally understandable", "that would be so frustrating", "super annoying",
            "you've done great", "you have done great", "you did great", "kudos",
            "valid concern", "i appreciate you sharing", "thanks for sharing"]
_CONTR = re.compile(r"\b(i'm|you're|we've|it's|don't|can't|that's|i'll|we're|there's|you'll|let's)\b")


def heuristic_style_score(text, prev_texts=()):
    """Cheap guardrail score: penalise AI tells / length / repeated openers, reward contractions."""
    t = text.lower()
    s = -0.6 * sum(t.count(x) for x in AI_TELLS)
    s += 0.25 * len(_CONTR.findall(t))
    w = text.split()
    s += 0.4 if len(w) <= 32 else -0.4 * ((len(w) - 32) / 20.0)
    if prev_texts:
        op = " ".join(w[:4]).lower()
        if any(" ".join(p.split()[:4]).lower() == op for p in prev_texts):
            s -= 1.0
    return s


class StyleReward(nn.Module):
    """Learned P(human) from the frozen-Gemma embedding of a reply (weights ship in the bundle)."""
    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


@torch.no_grad()
def best_of_n(gemma_lm, tok, prompt, gemma_feat, reward, n=6, device="cuda", exp_hidden=None,
              prev_texts=(), alpha=1.0, beta=0.3, temperature=1.0, max_new_tokens=110, return_all=False):
    """Sample n replies, score with the learned reward (+heuristic guardrails), keep the best."""
    from .gemma import generate_with_experience
    cands = [generate_with_experience(gemma_lm, tok, prompt, exp_hidden, device=device,
                                      max_new_tokens=max_new_tokens, temperature=temperature) for _ in range(n)]
    learned = torch.sigmoid(reward(gemma_feat.featurize(None, cands).float())).cpu().numpy()
    scored = [(alpha * float(learned[i]) + beta * heuristic_style_score(cands[i], prev_texts), cands[i])
              for i in range(n)]
    scored.sort(key=lambda x: -x[0])
    return (scored if return_all else scored[0][1])
