"""rl-sales-augment: a trained RL sales policy that augments ANY LLM.

The RL policy (bundled in this package as `data/rl_sales_agent.pt`) has learned, from a chaotic
multi-segment sales world, WHICH strategic move works next given the buyer's state. At serve time
it reads the conversation (perception), picks the move, and your LLM writes the words. The LLM is
pluggable via a `generate_fn(prompt) -> str`; the agent keeps its own memory (belief + history).

    import rl_sales_augment as rsa
    gen = rsa.providers.gemini_vertex(project="my-gcp-project")   # or openai_chat(), anthropic_chat(), ...
    bot = rsa.load_agent(gen, company_ctx="... your company facts ...")
    bot.new_conversation(segment=7)                 # optional: bias to a segment
    print(bot.reply("it feels expensive compared to just using AWS")["reply"])

The heavy bits (transformers/Gemma, provider SDKs, MCP) are optional extras; the core (this
import + `load_agent` + serving) needs only numpy + torch, so it runs on any Python that torch
supports. This distribution is serve-only: the world-model training pipeline and company
fine-tuning are the commercial offering of Convai Innovations Pvt. Ltd.
"""
from __future__ import annotations

__version__ = "0.6.0"

from .world import ACTION_NAMES, SEG_NAMES, SEGMENTS, SalesWorld, SalesConfig
from .deploy import AugmentedAgent, SalesBot, estimate_state_via, MOVE_INTENT, next_moves
from . import providers
from ._env import load_env

__all__ = [
    "load_agent", "load_gemma_bot", "model_path", "MODEL_PATH", "AugmentedAgent", "SalesBot",
    "estimate_state_via", "MOVE_INTENT", "next_moves", "providers",
    "ACTION_NAMES", "SEG_NAMES", "SEGMENTS", "SalesWorld", "SalesConfig", "load_env", "__version__",
]


def model_path() -> str:
    """Filesystem path to the bundled trained policy (`rl_sales_agent.pt`)."""
    from importlib.resources import files
    return str(files("rl_sales_augment").joinpath("data", "rl_sales_agent.pt"))


MODEL_PATH = model_path()


def load_agent(generate_fn, *, company_ctx: str = "", segment=None,
               rerank_n: int = 1, humanize: bool = True, device: str = "cpu") -> "AugmentedAgent":
    """Load the bundled policy and wrap ANY LLM (`generate_fn: prompt -> str`).

    company_ctx : block of your company's facts, injected into every reply prompt.
    segment     : optional segment id (0-9, see SEG_NAMES) to bias the internal world.
    rerank_n    : >1 enables best-of-N reranking of replies for naturalness.
    device      : the tiny policy runs fine on 'cpu'.
    """
    return AugmentedAgent(MODEL_PATH, generate_fn, device=device, company_ctx=company_ctx,
                          segment=segment, rerank_n=rerank_n, humanize=humanize)


def load_gemma_bot(model: str = "google/gemma-4-E2B-it", *, device: str = None, company_ctx: str = "",
                   segment=None, n_candidates: int = 1, perceive: bool = True) -> "SalesBot":
    """Gemma-native augmented bot (open-weights only) -> a `SalesBot` using Gemma 4 E2B plus the
    bundled policy, experience bridge, and trained style reranker.

    This path injects the RL 'experience' latent into Gemma's residual stream (the "common latent
    space"). Since v0.5.0 the bundled bridge is ALIGNED (trained via move-probe + reply
    self-distillation): with a neutral prompt, injection alone steers the reply toward the
    policy-chosen move (reply-executes-move 19% -> 49% on the training eval; 0% -> 58% in an
    independent local check; fluency unchanged). It is a complementary channel -- prompt-level move
    injection remains the primary mechanism. For a prompt-level bot that works with ANY LLM
    (including Gemma), use `load_agent(providers.gemma_e2b())` instead. Requires the [gemma] extra;
    `model` may be a local path or a Hugging Face repo id (auto-downloaded, not gated).
    """
    import torch
    from .gemma import GemmaFeaturizer
    if device is None:
        device = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    feat = GemmaFeaturizer(model, device=device, dtype=dtype)
    return SalesBot(MODEL_PATH, feat, device=device, company_ctx=company_ctx,
                    segment=segment, n_candidates=n_candidates, perceive=perceive)
