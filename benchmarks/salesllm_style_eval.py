"""SalesLLM-style benchmark harness -- faithful to the methodology of "Sell More, Play Less"
(arXiv 2604.07054): multi-turn persuasive sales dialogue with controllable buyer personas and
difficulty (cooperative -> adversarial), a CustomerLM-style buyer simulator kept strictly in the
BUYER role (their fix for the ~17% role-reversal problem), and a fully-automatic evaluation
pipeline = an LLM rater for sales-process progress + an end-of-dialogue buying-intent judge.

This REPLICATES the published methodology; it is NOT the official benchmark (whose code /
CustomerLM / BERT classifiers were not released as of mid-2026). It runs the SAME base LLM as the
seller WITH and WITHOUT the RL policy (rl-sales-augment), paired on identical scenarios, so the only
difference is who chooses the strategic move: the LLM itself vs the trained RL policy.

Usage:  python salesllm_style_eval.py [TURNS] [--project GCP_PROJECT]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rl_sales_augment as rsa
from rl_sales_augment._text import HUMAN_STYLE   # same style guidance for both arms (fairness)

# ---- scenarios: SalesLLM domains (Financial Services, Consumer Goods) + B2B tech ----
SCENARIOS = [
    {"name": "FinServ SaaS", "seg": 1, "product": "a portfolio-analytics SaaS for wealth advisors",
     "ctx": "You sell WealthLens, a portfolio-analytics SaaS for RIA wealth advisors, about $1,200 a month, "
            "integrates with Schwab and Fidelity, saves roughly 5 hours a week on client reporting."},
    {"name": "Retail hardware", "seg": 7, "product": "an in-store smart-shelf sensor system",
     "ctx": "You sell ShelfSense, in-store smart-shelf sensors, about $8k hardware plus $200 a month, "
            "cuts stockouts by around 20 percent and installs in a day."},
    {"name": "Dev tool", "seg": 2, "product": "a CI/CD security scanner",
     "ctx": "You sell BuildGuard, a CI/CD security scanner, about $4k a year, SOC2-ready, catches leaked "
            "secrets and vulnerabilities before deploy."},
    {"name": "Consumer goods", "seg": 0, "product": "a premium office coffee + machine service",
     "ctx": "You sell BeanFleet, premium office coffee with machine service, about $600 a month for a "
            "50-person office, next-day restocking."},
]

# ---- persona x difficulty (private disposition drives realistic, earned buying signals) ----
PERSONAS = [
    {"difficulty": "easy", "persona": "a warm, ready-to-move operations lead",
     "private": "you genuinely need this and have budget; you warm up quickly once the rep sounds competent."},
    {"difficulty": "medium", "persona": "a busy, ROI-skeptical manager",
     "private": "you have a real pain but doubt the ROI; you only open up if the rep ties value to your problem."},
    {"difficulty": "hard", "persona": "an adversarial, price-focused CFO",
     "private": "you distrust vendors, push hard on price, and only move if the rep handles objections and proves ROI."},
]

OPENER = "Hi, thanks for hopping on. What prompted you to look into this?"


def _chat(transcript, me):
    """Render the transcript from `me`'s perspective (my turns = assistant, other = user)."""
    return [{"role": "assistant" if spk == me else "user", "content": txt} for spk, txt in transcript]


def buyer_turn(gen, scenario, pcfg, transcript):
    system = (f"You are the BUYER on a sales call about {scenario['product']}. "
              f"Persona: {pcfg['persona']}. Your PRIVATE disposition (never state it outright): {pcfg['private']} "
              "Stay strictly in the BUYER role: never act as the salesperson, never pitch or offer to sell. "
              "Reply like a real person in 1-2 short sentences, consistent with your disposition. Reveal buying "
              "interest only if the rep genuinely earns it. No em dashes.")
    return gen(system=system, history=_chat(transcript, "buyer")).strip()


def base_seller_turn(gen, scenario, transcript):
    system = (f"{scenario['ctx']}\n\nYou are a skilled human sales rep. Advance the sale effectively using the "
              f"company facts and the conversation. {HUMAN_STYLE} Keep it to 1-2 short sentences.")
    return gen(system=system, history=_chat(transcript, "seller")).strip()


def run_dialogue(gen, scenario, pcfg, arm, turns):
    transcript = [("seller", OPENER)]
    bot = None
    if arm == "rl":
        bot = rsa.load_agent(gen, company_ctx=scenario["ctx"])
        bot.new_conversation(segment=scenario["seg"])
    for _ in range(turns):
        buyer_msg = buyer_turn(gen, scenario, pcfg, transcript)
        transcript.append(("buyer", buyer_msg))
        seller_msg = bot.reply(buyer_msg)["reply"] if arm == "rl" else base_seller_turn(gen, scenario, transcript)
        transcript.append(("seller", seller_msg))
    return transcript


def judge(gen, scenario, transcript):
    convo = "\n".join(f"{'Rep' if s == 'seller' else 'Buyer'}: {t}" for s, t in transcript)
    prompt = (f"You are an expert sales evaluator. Read this conversation about {scenario['product']}.\n\n"
              f"{convo}\n\nRate it. Return ONLY JSON like "
              '{"process_progress": 0.0, "buying_intent": 0.0, "decision": "no"} where process_progress is '
              "0.0-1.0 for how far the sale advanced through rapport, discovery, value, objection-handling, close; "
              "buying_intent is 0.0-1.0 for the buyer's end-of-dialogue intent to purchase; decision is buy or no.")
    txt = gen(prompt)
    try:
        import re
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        return {"process_progress": float(d.get("process_progress", 0)),
                "buying_intent": float(d.get("buying_intent", 0)),
                "buy": str(d.get("decision", "no")).lower().startswith("buy")}
    except Exception:
        return {"process_progress": 0.0, "buying_intent": 0.0, "buy": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("turns", nargs="?", type=int, default=7)
    ap.add_argument("--project", default="nextjslms")
    args = ap.parse_args()
    gen = rsa.providers.gemini_vertex(project=args.project)

    records, t0 = [], time.time()
    for sc in SCENARIOS:
        for pc in PERSONAS:
            row = {"scenario": sc["name"], "difficulty": pc["difficulty"]}
            for arm in ("base", "rl"):
                tr = run_dialogue(gen, sc, pc, arm, args.turns)
                row[arm] = judge(gen, sc, tr)
                row[arm]["transcript"] = [{"speaker": s, "text": t} for s, t in tr]
            records.append(row)
            b, r = row["base"], row["rl"]
            print(f"  {sc['name']:16} {pc['difficulty']:6} | base buy={b['buy']} int={b['buying_intent']:.2f} "
                  f"| rl buy={r['buy']} int={r['buying_intent']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate ----
    diffs = ["easy", "medium", "hard"]
    agg = {arm: {"buy": [], "prog": [], "int": [], "by_diff": {d: {"buy": [], "prog": []} for d in diffs}}
           for arm in ("base", "rl")}
    for row in records:
        for arm in ("base", "rl"):
            e = row[arm]
            agg[arm]["buy"].append(e["buy"]); agg[arm]["prog"].append(e["process_progress"]); agg[arm]["int"].append(e["buying_intent"])
            agg[arm]["by_diff"][row["difficulty"]]["buy"].append(e["buy"])
            agg[arm]["by_diff"][row["difficulty"]]["prog"].append(e["process_progress"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    print("\n" + "=" * 66)
    print(f"{'arm':6} | conversion | avg process-progress | avg buying-intent")
    for arm in ("base", "rl"):
        a = agg[arm]
        print(f"{arm:6} |   {mean(a['buy']):.0%}     |        {mean(a['prog']):.2f}          |     {mean(a['int']):.2f}")

    out = {"n_dialogues_per_arm": len(records), "turns": args.turns,
           "summary": {arm: {"conversion": mean(agg[arm]["buy"]), "process_progress": mean(agg[arm]["prog"]),
                             "buying_intent": mean(agg[arm]["int"]),
                             "by_difficulty": {d: {"conversion": mean(agg[arm]["by_diff"][d]["buy"]),
                                                   "process_progress": mean(agg[arm]["by_diff"][d]["prog"])}
                                               for d in diffs}} for arm in ("base", "rl")},
           "records": records}
    here = os.path.dirname(__file__)
    json.dump(out, open(os.path.join(here, "salesllm_results.json"), "w"), indent=1)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(diffs) + 1)  # easy/medium/hard + overall
        conv = {arm: [mean(agg[arm]["by_diff"][d]["buy"]) for d in diffs] + [mean(agg[arm]["buy"])] for arm in ("base", "rl")}
        prog = {arm: [mean(agg[arm]["by_diff"][d]["prog"]) for d in diffs] + [mean(agg[arm]["prog"])] for arm in ("base", "rl")}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5)); w = 0.38
        for ax, data, title in [(ax1, conv, "Buying-intent conversion rate"),
                                (ax2, prog, "Sales-process progress (0-1)")]:
            ax.bar(x - w/2, data["base"], w, label="base LLM", color="#9aa0a6")
            ax.bar(x + w/2, data["rl"], w, label="+ RL policy (ours)", color="#1a73e8")
            ax.set_xticks(x); ax.set_xticklabels(diffs + ["overall"]); ax.set_title(title)
            ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=0.3)
            for i, (bb, rr) in enumerate(zip(data["base"], data["rl"])):
                ax.text(x[i]-w/2, bb+0.02, f"{bb:.0%}" if "conversion" in title else f"{bb:.2f}", ha="center", fontsize=8)
                ax.text(x[i]+w/2, rr+0.02, f"{rr:.0%}" if "conversion" in title else f"{rr:.2f}", ha="center", fontsize=8)
        fig.suptitle(f"SalesLLM-style benchmark: base LLM vs RL-augmented (Gemini seller, {len(records)} paired dialogues)")
        fig.tight_layout()
        fig.savefig(os.path.join(here, "salesllm_results.png"), dpi=130)
        print("\nsaved salesllm_results.png + salesllm_results.json")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
