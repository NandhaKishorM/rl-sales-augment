"""Serve-time state model of the B2B sales world.

This module carries what the trained policy needs at inference time: the segment
taxonomy (each deal type has its own economics), the deal/lead state container, and
the observation / text rendering the policy was trained on. The agent's perception
step writes the perceived buyer state onto a `Lead`; the policy reads `_obs()` /
`describe()` and picks the move.

NOTE: the full world model -- the market dynamics, customer-response physics,
close-probability model, shocks, and the training pipeline built on them -- is NOT
part of this distribution. That is the proprietary training stack of
Convai Innovations Pvt. Ltd.; contact nandakishor@convaiinnovations.com for
training, retraining, or company fine-tuning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# sales moves
RESEARCH, RAPPORT, PITCH, HANDLE_OBJECTION, DISCOUNT, FOLLOW_UP, CLOSE, DROP = range(8)
ACTION_NAMES = ["RESEARCH", "RAPPORT", "PITCH", "OBJECTION", "DISCOUNT", "FOLLOWUP", "CLOSE", "DROP"]
N_ACTIONS = len(ACTION_NAMES)

# Extensive deal-type taxonomy across software + hardware. Each segment has its OWN
# economics -> the optimal play is segment-specific (enterprise needs long trust and
# resists discounts; SMB SaaS closes fast; hardware is price-driven). Columns:
#   value($k), difficulty, churn_mult, trust_need(before pitch), discount_effectiveness
SEGMENTS = [
    ("SaaS SMB (self-serve)",         0.8, 0.00, 1.4, 0.30, 1.3),
    ("SaaS Mid-Market",               2.0, 0.10, 1.0, 0.45, 1.0),
    ("SaaS Enterprise",               6.0, 0.28, 0.6, 0.65, 0.6),
    ("Dev Tools / API (bottom-up)",   1.0, 0.05, 1.2, 0.30, 1.1),
    ("Security / Compliance SW",      4.0, 0.24, 0.7, 0.70, 0.8),
    ("Perpetual License Software",    2.5, 0.12, 0.9, 0.50, 1.0),
    ("Hardware Device (goods)",       1.2, 0.05, 1.1, 0.30, 1.5),
    ("Hardware Infra / Datacenter",   8.0, 0.32, 0.5, 0.68, 0.5),
    ("IoT / Embedded (HW+SW)",        3.0, 0.20, 0.8, 0.55, 0.9),
    ("Managed Services / MSP",        3.5, 0.18, 0.7, 0.60, 0.85),
]
SEG_NAMES = [s[0] for s in SEGMENTS]
SEG_VALUE = np.array([s[1] for s in SEGMENTS])
SEG_DIFF  = np.array([s[2] for s in SEGMENTS])
SEG_CHURN = np.array([s[3] for s in SEGMENTS])
SEG_TRUST = np.array([s[4] for s in SEGMENTS])
SEG_DISC  = np.array([s[5] for s in SEGMENTS])
N_SEGMENTS = len(SEGMENTS)

PER_LEAD_OBS = 5 + 6   # segment economics(5) + interest,trust,budget,objection,patience,est_fit
GLOBAL_OBS = 5         # market_heat, competition, reputation, discount_budget, step


@dataclass
class SalesConfig:
    """Serve-time configuration. The training-time physics parameters are not distributed."""
    n_leads: int = 4
    horizon: int = 60                 # a sales quarter
    segment_ids: tuple = None         # restrict leads to these segments (company focus); None=all
    init_discount_budget: float = 6.0


class Lead:
    def __init__(self, rng, cfg: SalesConfig):
        self.cfg = cfg
        if cfg.segment_ids:
            self.seg = int(cfg.segment_ids[int(rng.integers(0, len(cfg.segment_ids)))])
        else:
            self.seg = int(rng.integers(0, N_SEGMENTS))
        self.value = float(SEG_VALUE[self.seg]); self.difficulty = float(SEG_DIFF[self.seg])
        self.churn_mult = float(SEG_CHURN[self.seg]); self.trust_need = float(SEG_TRUST[self.seg])
        self.disc_effect = float(SEG_DISC[self.seg])
        self.true_fit = float(rng.random())            # HIDDEN quality of the match
        self.interest = float(np.clip(0.15 + 0.15 * rng.random(), 0, 1))
        self.trust = float(np.clip(0.10 + 0.15 * rng.random(), 0, 1))
        self.budget_fit = float(np.clip(0.2 + 0.4 * rng.random(), 0, 1))
        self.objection = float(np.clip(0.2 + 0.25 * rng.random(), 0, 1))
        self.patience = 1.0
        self.discount_given = 0.0
        self.est_fit = 0.5                              # observable estimate of true_fit

    def obs(self):
        seg = [self.value / 8.0, self.difficulty, self.churn_mult / 1.5,
               self.trust_need, self.disc_effect / 1.5]   # segment economics (observable)
        return np.array(seg + [self.interest, self.trust, self.budget_fit,
                               self.objection, self.patience, self.est_fit], np.float32)


class SalesWorld:
    """Serve-time belief state: holds the deals and renders the exact observation /
    description the policy was trained on. Perception updates the lead fields; the
    dynamics (`step`) are part of the private training stack and are not included."""
    n_actions = N_ACTIONS

    def __init__(self, config: Optional[SalesConfig] = None, seed: Optional[int] = None):
        self.cfg = config or SalesConfig()
        self.n = self.cfg.n_leads
        self.rng = np.random.default_rng(seed)
        self._reset_state()

    @property
    def action_nvec(self):
        return [N_ACTIONS] * self.n

    @property
    def obs_dim(self):
        return self.n * PER_LEAD_OBS + GLOBAL_OBS

    def _reset_state(self):
        self.leads: List[Lead] = [Lead(self.rng, self.cfg) for _ in range(self.n)]
        self.market_heat = 0.55
        self.competition = 0.4
        self.reputation = 0.7
        self.discount_budget = self.cfg.init_discount_budget
        self.step_i = 0
        self.revenue_total = 0.0
        self.deals_won = 0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), self._info()

    # -------------------------------------------------- obs / info / text
    def _obs(self):
        c = self.cfg
        parts = [l.obs() for l in self.leads]
        glob = np.array([self.market_heat, self.competition, self.reputation,
                         min(self.discount_budget / c.init_discount_budget, 1.0),
                         self.step_i / c.horizon], np.float32)
        return np.concatenate(parts + [glob]).astype(np.float32)

    def _info(self):
        return {"revenue_total": self.revenue_total, "deals_won": self.deals_won,
                "step": self.step_i, "reputation": self.reputation}

    def describe(self) -> str:
        c = self.cfg
        heat = ("hot" if self.market_heat > 0.65 else "cooling" if self.market_heat < 0.4 else "steady")
        comp = ("fierce" if self.competition > 0.6 else "moderate" if self.competition > 0.35 else "light")
        lines = [f"Sales quarter, week {self.step_i+1}/{c.horizon}. Market demand is {heat}; "
                 f"competition is {comp}; brand reputation {self.reputation:.0%}; "
                 f"discount budget {min(self.discount_budget/c.init_discount_budget,1):.0%} left."]
        for i, l in enumerate(self.leads):
            def w(x, lo, hi, names): return names[0] if x < lo else names[2] if x > hi else names[1]
            lines.append(
                f"Deal {i+1} ({SEG_NAMES[l.seg]}, ~${l.value:.1f}k): "
                f"interest {w(l.interest,.35,.65,['low','medium','high'])}, "
                f"trust {w(l.trust,.35,.65,['low','medium','high'])}, "
                f"budget-fit {w(l.budget_fit,.35,.65,['weak','ok','strong'])}, "
                f"objections {w(l.objection,.3,.6,['few','some','many'])}, "
                f"{'going cold' if l.patience<0.35 else 'engaged'}"
                f"{', already discounted' if l.discount_given>0 else ''}.")
        return " ".join(lines)
