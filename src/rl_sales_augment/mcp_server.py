"""MCP server that exposes the trained RL sales policy as tools.

The connecting client's LLM (Claude Desktop, Cursor, Windsurf, ...) does the perception and writes
the words; this server supplies the STRATEGY: given a perceived buyer state it returns the RL-chosen
move. The policy is tiny and runs on CPU, so no LLM is needed on the server side.

Run (stdio, for local MCP clients):   rl-sales-augment-mcp
Or:                                    python -m rl_sales_augment.mcp_server
HTTP:                                  rl-sales-augment-mcp streamable-http

Needs the [mcp] extra:  pip install "rl-sales-augment[mcp]"
"""
from __future__ import annotations
from typing import Optional

try:                                      # standalone fastmcp (3.x), the community standard
    from fastmcp import FastMCP
except ImportError:                       # bundled in the official `mcp` SDK
    from mcp.server.fastmcp import FastMCP

from .world import ACTION_NAMES, SEG_NAMES
from .deploy import MOVE_INTENT

_INSTRUCTIONS = (
    "Augment your sales replies with a trained reinforcement-learning policy. Workflow: "
    "(1) read the conversation and estimate the buyer's state (or call perception_prompt for the "
    "exact rubric); (2) call next_move with the five 0-1 numbers to get the RL-chosen strategic "
    "move and what it should accomplish; (3) write your reply carrying out that move, grounded in "
    "your own company facts. The policy learned which move works from a chaotic sales world; you "
    "supply the words."
)

mcp = FastMCP("rl-sales-augment", instructions=_INSTRUCTIONS)

_agent = None


def _compute_move(interest, trust, budget_fit, objection, patience, segment):
    """Run the tiny RL policy on a perceived buyer state -> chosen move index. LLM-free."""
    global _agent
    if _agent is None:
        from . import load_agent
        _agent = load_agent(generate_fn=lambda p: "{}")   # dummy LLM: we only use the RL policy
    _agent.new_conversation(segment=segment)
    lead = _agent.world.leads[0]
    belief = dict(interest=interest, trust=trust, budget_fit=budget_fit,
                  objection=objection, patience=patience)
    for k, v in belief.items():
        setattr(lead, k, float(min(1.0, max(0.0, v))))
    return _agent._rl_move(), belief


@mcp.tool()
def next_move(interest: float, trust: float, budget_fit: float, objection: float,
              patience: float, segment: Optional[int] = None) -> dict:
    """Return the RL-chosen next sales move for a perceived buyer state.

    Each argument is 0.0-1.0: interest (how keen), trust, budget_fit (how affordable it feels),
    objection (unresolved concerns; 0=none, 1=many), patience (1=engaged, 0=about to walk).
    segment is an optional market-segment id 0-9 (see list_segments). Returns the strategic move
    (RAPPORT / PITCH / OBJECTION / DISCOUNT / CLOSE / FOLLOWUP / RESEARCH / DROP), what that move
    should accomplish, and the belief echoed back. You then write the reply that executes the move.
    """
    idx, belief = _compute_move(interest, trust, budget_fit, objection, patience, segment)
    return {"chosen_move": ACTION_NAMES[idx], "intent": MOVE_INTENT[idx],
            "belief": belief, "segment": segment}


@mcp.tool()
def perception_prompt(conversation: str) -> str:
    """Return the rubric for estimating the buyer's state from a conversation. Answer it yourself to
    get the five 0-1 numbers, then call next_move with them."""
    return ("Rate the BUYER'S CURRENT state as five numbers from 0.0 to 1.0: interest, trust, "
            "budget_fit, objection (0=none, 1=many), patience (1=engaged, 0=about to walk). "
            "Then call next_move with those five numbers.\n\nCONVERSATION:\n" + conversation)


@mcp.tool()
def list_moves() -> dict:
    """The strategic moves the policy can choose, each mapped to what it should accomplish."""
    return {ACTION_NAMES[i]: MOVE_INTENT[i] for i in range(len(ACTION_NAMES))}


@mcp.tool()
def list_segments() -> dict:
    """The market segments (id -> name) the policy is aware of."""
    return {i: name for i, name in enumerate(SEG_NAMES)}


def main():
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
