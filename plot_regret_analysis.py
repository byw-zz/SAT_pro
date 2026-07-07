"""Regret analysis figure for the OAT sensitivity experiment (reviewer-facing).

Three grouped-bar panels over the 4 datasets x 4 perturbed parameters:
  A  mean |regret|   -- typical optimality loss (near zero everywhere)
  B  max  |regret|   -- worst case (only structured nP12-18 spikes)
  C  mean Jaccard    -- strategy stability; explains that the structured spike
                        is rare strategy flips, not systematic sub-optimality.
Palette: dataviz categorical slots 1-4 (CVD-safe by the reference ordering).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
PARAMS = ["C_benefit", "P_loss", "D_cost", "E_prob"]
PCOL = {"C_benefit": "#c6c6c6", "P_loss": "#fccdba", "D_cost": "#c2c4e3", "E_prob": "#dac8e3"}
GROUPS = [
    ("structured_12_18", "structured\nnP12-18"),
    ("structured_151_199", "structured\nnP151-199"),
    ("random_nP10", "random\nnP10"),
    ("random_nP100", "random\nnP100"),
]
glabs = [l for _, l in GROUPS]

frames = []
for d, lab in GROUPS:
    f = pd.read_csv(f"sensitivity_out/{d}/sensitivity_results.csv")
    f["group"] = lab
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df["absreg"] = df["optimality_regret"].abs()

agg = df.groupby(["group", "param"]).agg(
    mean_reg=("absreg", "mean"), max_reg=("absreg", "max"), mean_jac=("jaccard", "mean")
).reset_index()

def val(g, p, key):
    r = agg[(agg.group == g) & (agg.param == p)]
    return float(r[key].iloc[0]) if len(r) else 0.0

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "font.family": "sans-serif",
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(len(glabs))
w = 0.19
offs = {p: (i - 1.5) * w for i, p in enumerate(PARAMS)}

def grouped(ax, key, log, floor=None):
    for p in PARAMS:
        vals = [val(g, p, key) for g in glabs]
        base = floor if (log and floor) else 0
        heights = [max(v, floor) if (log and floor) else v for v in vals]
        ax.bar(x + offs[p], heights, w, bottom=(base if not log else None),
               color=PCOL[p], label=p, edgecolor="#52514e", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(glabs)
    ax.grid(False, which="both")
    if log:
        ax.set_yscale("log")

grouped(axA, "mean_reg", log=True, floor=1e-4)
axA.set_ylabel("mean |regret|  (log)")
axA.set_title("A  Mean |regret|: typical loss ~ 0")
axA.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

grouped(axB, "max_reg", log=True, floor=1e-4)
axB.set_ylabel("max |regret|  (log)")
axB.set_title("B  Max |regret|: worst case only for structured nP12-18")

fig.suptitle("Robustness of the MaxSAT defense strategy to +/-10/20/30% parameter perturbation",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "sensitivity_out/regret_analysis.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

print(f"\n{'group':<22}{'param':<11}{'mean|reg|':>10}{'max|reg|':>10}{'meanJac':>9}")
for g in glabs:
    for p in PARAMS:
        print(f"{g.replace(chr(10),' '):<22}{p:<11}{val(g,p,'mean_reg'):>10.4f}{val(g,p,'max_reg'):>10.3f}{val(g,p,'mean_jac'):>9.3f}")
