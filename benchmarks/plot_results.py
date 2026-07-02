"""Regenerate salesllm_results.png from salesllm_results.json (no benchmark re-run)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(__file__)
s = json.load(open(os.path.join(here, "salesllm_results.json")))
summ, n = s["summary"], s["n_dialogues_per_arm"]
diffs = ["easy", "medium", "hard"]
x = np.arange(len(diffs) + 1)


def series(arm, key):
    return [summ[arm]["by_difficulty"][d][key] for d in diffs] + [summ[arm][key.replace("process_progress", "process_progress")]]


conv = {a: [summ[a]["by_difficulty"][d]["conversion"] for d in diffs] + [summ[a]["conversion"]] for a in ("base", "rl")}
prog = {a: [summ[a]["by_difficulty"][d]["process_progress"] for d in diffs] + [summ[a]["process_progress"]] for a in ("base", "rl")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
w = 0.38
for ax, data, title, pct in [(ax1, conv, "Buying-intent conversion rate", True),
                             (ax2, prog, "Sales-process progress (0-1)", False)]:
    ax.bar(x - w/2, data["base"], w, label="base LLM", color="#9aa0a6")
    ax.bar(x + w/2, data["rl"], w, label="+ RL policy (ours)", color="#1a73e8")
    ax.set_xticks(x); ax.set_xticklabels(diffs + ["overall"])
    ax.set_title(title, pad=10); ax.set_ylim(0, 1.18); ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i in range(len(x)):
        for off, arm in [(-w/2, "base"), (w/2, "rl")]:
            v = data[arm][i]
            ax.text(x[i] + off, v + 0.02, f"{v:.0%}" if pct else f"{v:.2f}", ha="center", fontsize=8)
fig.suptitle(f"SalesLLM-style benchmark: base LLM vs RL-augmented (Gemini seller, {n} paired dialogues)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(here, "salesllm_results.png"), dpi=130, bbox_inches="tight")
print("wrote salesllm_results.png")
