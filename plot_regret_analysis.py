"""Regret analysis figures for the OAT sensitivity experiment.

Reads the four sensitivity_results.csv files and renders a 2x2 figure that
characterizes optimality-regret: its relation to strategy overlap, the
mean-vs-median inflation, the scaling trend, and the per-group distribution.
Palette: dataviz categorical slots 1-4 (CVD-safe by the reference ordering).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- dataviz light-mode tokens ---
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# categorical slots 1-4 -> one per perturbed parameter (fixed order)
PCOL = {"C_benefit": "#2a78d6", "P_loss": "#1baf7a", "D_cost": "#eda100", "E_prob": "#008300"}
BLUE_FILL, BLUE_LINE, RED = "#cde2fb", "#184f95", "#e34948"

GROUPS = [
    ("structured_12_18", "structured\nnP12-18"),
    ("structured_151_199", "structured\nnP151-199"),
    ("random_nP10", "random\nnP10"),
    ("random_nP100", "random\nnP100"),
]

frames = []
for d, lab in GROUPS:
    f = pd.read_csv(f"sensitivity_out/{d}/sensitivity_results.csv")
    f["group"] = lab
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df["absreg"] = df["optimality_regret"].abs()
labs = [l for _, l in GROUPS]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "font.family": "sans-serif",
})

fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(12, 9))

# A: regret vs jaccard (symlog y), colored by param
for p, c in PCOL.items():
    s = df[df.param == p]
    axA.scatter(s.jaccard, s.optimality_regret, s=9, c=c, alpha=0.35,
                edgecolors="none", label=p)
axA.set_yscale("symlog", linthresh=0.05)
axA.axhline(0, color=MUTED, lw=0.8)
axA.set_xlabel("Jaccard(D* perturbed, D* baseline)")
axA.set_ylabel("optimality regret  (symlog)")
axA.set_title("A  Large |regret| appears only when Jaccard < 1")
axA.grid(axis="y", color=GRID, lw=0.6)
axA.legend(frameon=False, fontsize=8, loc="upper left")

# B: mean vs median |regret| per group (mean inflated by outliers)
means = [df[df.group == l].absreg.mean() for l in labs]
meds = [df[df.group == l].absreg.median() for l in labs]
x = np.arange(len(labs))
for xi, (m, md) in enumerate(zip(means, meds)):
    axB.plot([xi, xi], [md, m], color=MUTED, lw=1.2, zorder=1)
axB.scatter(x, means, s=70, c=RED, zorder=3, label="mean |regret|")
axB.scatter(x, meds, s=70, c=BLUE_LINE, zorder=3, label="median |regret|")
axB.set_xticks(x); axB.set_xticklabels(labs)
axB.set_ylabel("|regret|")
for xi, m in enumerate(means):
    axB.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(9, 0),
                 fontsize=7, color=RED, va="center", family="monospace")
axB.set_title("B  Mean is inflated by rare outliers; median stays ~ 0")
axB.grid(axis="y", color=GRID, lw=0.6)
axB.legend(frameon=False, fontsize=8)

# C: median |regret| vs scale, by param (structured nP12-18)
sub = df[df.group == "structured\nnP12-18"]
scales = sorted(sub.scale.unique())
for p, c in PCOL.items():
    mn = [sub[(sub.param == p) & (sub.scale == s)].absreg.mean() for s in scales]
    axC.plot(scales, mn, "-o", color=c, ms=5, lw=2, label=p)
axC.axvline(1.0, color=MUTED, lw=0.8, ls=":")
axC.set_xlabel("parameter scale factor")
axC.set_ylabel("mean |regret|")
axC.set_title("C  Mean |regret| vs scaling (structured nP12-18): peaks at -30% risk params")
axC.grid(axis="y", color=GRID, lw=0.6)
axC.legend(frameon=False, fontsize=8)

# D: |regret| distribution per group (log y)
data = [df[df.group == l].absreg.replace(0, 1e-4).values for l in labs]
bp = axD.boxplot(data, tick_labels=labs, showfliers=True, patch_artist=True,
                 widths=0.55,
                 flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none", alpha=0.35),
                 medianprops=dict(color=BLUE_LINE, lw=1.5),
                 whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for b in bp["boxes"]:
    b.set(facecolor=BLUE_FILL, edgecolor=BLUE_LINE, lw=1.0)
axD.set_yscale("log")
axD.set_ylabel("|regret|  (log)")
axD.set_title("D  |regret| distribution: mass near 0, rare high outliers")
axD.grid(axis="y", color=GRID, lw=0.6)

fig.suptitle("Optimality-regret analysis  (OAT sensitivity, +/-10/20/30% multiplicative scaling)",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "sensitivity_out/regret_analysis.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

print("\nper-group regret stats:")
for l in labs:
    s = df[df.group == l]
    print(f"  {l.replace(chr(10),' '):<22} mean={s.absreg.mean():.3f}  median={s.absreg.median():.4f}  "
          f"p90={s.absreg.quantile(0.9):.3f}  max={s.absreg.max():.3f}  neg_regret_frac={(s.optimality_regret<0).mean():.2f}")
