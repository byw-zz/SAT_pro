"""Regret analysis figure for the OAT sensitivity experiment (reviewer-facing).

Three grouped-bar panels over the 4 datasets x 4 perturbed parameters:
  A  mean r_s        -- typical optimality loss (near zero everywhere)
  B  maximum r_s     -- worst case (only structured small spikes)
  C  mean Jaccard    -- strategy stability; explains that the structured spike
                        is rare strategy flips, not systematic sub-optimality.
Palette: dataviz categorical slots 1-4 (CVD-safe by the reference ordering).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
PARAMS = ["C_benefit", "P_loss", "D_cost", "E_prob"]
PCOL = {"C_benefit": "#c6c6c6", "P_loss": "#fccdba", "D_cost": "#c2c4e3", "E_prob": "#dac8e3"}
GROUPS = [
    ("structured_12_18", "structured\nsmall"),
    ("structured_151_199", "structured\nmedium"),
    ("random_nP10", "random\nsmall"),
    ("random_nP100", "random\nmedium"),
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

def grouped(ax, key, log):
    series = {p: [val(g, p, key) for g in glabs] for p in PARAMS}
    positive = [v for vals in series.values() for v in vals if v > 0]
    baseline = 10 ** (np.floor(np.log10(min(positive))) - 1) if log and positive else 0.0

    for p in PARAMS:
        vals = series[p]
        xpos = x + offs[p]
        positive_idx = [i for i, v in enumerate(vals) if v > 0]
        if positive_idx:
            pos_x = [xpos[i] for i in positive_idx]
            pos_vals = [vals[i] for i in positive_idx]
            heights = [v - baseline for v in pos_vals] if log else pos_vals
            ax.bar(pos_x, heights, w, bottom=baseline if log else 0,
                   color=PCOL[p], label=p, edgecolor="#52514e", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(glabs)
    ax.grid(False, which="both")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=baseline)

grouped(axA, "mean_reg", log=True)
axA.set_ylabel("Absolute normalized regret (log scale)")
axA.set_title(r"A  Mean $r_s$")
axA.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

grouped(axB, "max_reg", log=True)
axB.set_ylabel("Absolute normalized regret (log scale)")
axB.set_title(r"B  Maximum $r_s$")

fig.tight_layout()
out_png = "sensitivity_out/regret_analysis.png"
out_pdf = "sensitivity_out/regret_analysis.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print("saved", out_png)
print("saved", out_pdf)

print(f"\n{'group':<22}{'param':<11}{'mean|reg|':>10}{'max|reg|':>10}{'meanJac':>9}")
for g in glabs:
    for p in PARAMS:
        print(f"{g.replace(chr(10),' '):<22}{p:<11}{val(g,p,'mean_reg'):>10.4f}{val(g,p,'max_reg'):>10.3f}{val(g,p,'mean_jac'):>9.3f}")
