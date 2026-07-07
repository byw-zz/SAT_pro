"""Three-way comparison figure: MaxSAT (exact) vs GA-BP vs Khouzani-MILP baseline.

Reads the per-graph Khouzani results (comparison/khouzani_out_*.json, each scored
under the dataset's native evaluator: VE for small, BP for large) and the stored
GA/MaxSAT results (the four *_{n}graphs.json datasets). Produces two panels:

  A. Defence quality vs. the exact MaxSAT method — for each dataset, GA and
     Khouzani as 100%-stacked win/tie/loss bars (from the baseline's viewpoint).
  B. Solve time per method per dataset (log scale).

The story: on small graphs the baselines stay competitive with MaxSAT (GA more so);
at scale MaxSAT dominates in both quality (baselines lose every instance) and speed.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataset order: (label, khouzani_out file, stored dataset file)
DSETS = [
    ("structured\nsmall [12,17]", "comparison/khouzani_out_s_small.json", "structured_12_17_100graphs.json"),
    ("random\nsmall (nP=10)",     "comparison/khouzani_out_r_small.json", "random_10_100graphs.json"),
    ("structured\nlarge [151,199]","comparison/khouzani_out_s_large.json","structured_151_199_10graphs.json"),
    ("random\nlarge (nP=100)",    "comparison/khouzani_out_r_large.json", "random_100_10graphs.json"),
]

# palette (teal / yellow / coral)
INK   = "#0b0b0b"; MUTED = "#898781"; SURF = "#fcfcfb"
TEAL  = "#299d8f"; YELLOW = "#e9c46a"; CORAL = "#d87659"
WIN   = TEAL       # baseline better (good)
TIE   = YELLOW     # tie (neutral)
LOSS  = CORAL      # MaxSAT better (baseline loses)
C_SAT = TEAL; C_GA = CORAL; C_KH = YELLOW               # methods: MaxSAT / GA / Khouzani


def _wtl(rows, key, tol_abs=0.5, tol_rel=1e-3):
    """Khouzani win/tie/loss vs rows[*][key] (higher obj = better)."""
    w = t = l = 0
    for r in rows:
        a, b = r["khouzani_obj"], r.get(key)
        if a is None or b is None:
            continue
        tol = max(tol_abs, tol_rel * abs(b))
        if a > b + tol: w += 1
        elif a < b - tol: l += 1
        else: t += 1
    return w, t, l


def load():
    out = []
    for label, khf, dsf in DSETS:
        kh = json.load(open(os.path.join(ROOT, khf)))["rows"]
        ds = json.load(open(os.path.join(ROOT, dsf)))
        s = ds["summary"]
        n = len(kh)
        # Khouzani vs MaxSAT / vs GA
        kh_v_sat = _wtl(kh, "maxsat_obj")
        kh_v_ga  = _wtl(kh, "ga_obj")
        # GA vs MaxSAT (from stored summary; GA-win / tie / SAT-win)
        ga_v_sat = (s.get("num_ga_better", 0), s.get("num_tie", 0), s.get("num_sat_better", 0))
        # runtimes
        t_sat = s.get("avg_maxsat_time_s")
        t_ga  = s.get("avg_bp_time_s")
        t_kh  = sum(r["khouzani_milp_ms"] for r in kh) / len(kh) / 1000.0
        out.append({"label": label, "n": n,
                    "kh_v_sat": kh_v_sat, "kh_v_ga": kh_v_ga, "ga_v_sat": ga_v_sat,
                    "t_sat": t_sat, "t_ga": t_ga, "t_kh": t_kh})
    return out


def pct(triple):
    tot = sum(triple) or 1
    return [100.0 * x / tot for x in triple]


def main():
    data = load()
    labels = [d["label"] for d in data]
    x = list(range(len(data)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    fig.patch.set_facecolor(SURF)

    # ---- Panel A: quality vs MaxSAT (GA and Khouzani), 100% stacked W/T/L ----
    axA.set_facecolor(SURF)
    bw = 0.36
    for off, meth, keyname in [(-bw/2 - 0.01, "GA", "ga_v_sat"), (bw/2 + 0.01, "Khouzani", "kh_v_sat")]:
        for i, d in enumerate(data):
            win, tie, loss = pct(d[keyname])            # baseline-win / tie / SAT-win
            xi = x[i] + off
            axA.bar(xi, win, bw, bottom=0, color=WIN, edgecolor=SURF, linewidth=1.2)
            axA.bar(xi, tie, bw, bottom=win, color=TIE, edgecolor=SURF, linewidth=1.2)
            axA.bar(xi, loss, bw, bottom=win + tie, color=LOSS, edgecolor=SURF, linewidth=1.2)
            # method tag under each bar
            axA.text(xi, -4, meth, ha="center", va="top", fontsize=8, color=MUTED, rotation=0)
    axA.set_ylim(0, 100)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9, color=INK)
    axA.tick_params(axis="x", length=0, pad=22)
    axA.set_ylabel("share of graphs (%)", fontsize=10, color=INK)
    axA.set_title("Baseline vs. exact MaxSAT   (win / tie / loss)", fontsize=11, color=INK, pad=8)
    for s_ in ("top", "right", "left"):
        axA.spines[s_].set_visible(False)
    axA.spines["bottom"].set_color(MUTED)
    axA.tick_params(colors=MUTED)
    legA = [Patch(facecolor=WIN, label="baseline wins"),
            Patch(facecolor=TIE, label="tie (same defence)"),
            Patch(facecolor=LOSS, label="MaxSAT wins")]
    axA.legend(handles=legA, loc="lower center", bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=8.5, handlelength=1.1)

    # ---- Panel B: solve time per method (log scale) ----
    axB.set_facecolor(SURF)
    gw = 0.26
    series = [("MaxSAT", C_SAT, "t_sat"), ("GA-BP", C_GA, "t_ga"), ("Khouzani", C_KH, "t_kh")]
    for j, (name, col, key) in enumerate(series):
        xs = [xi + (j - 1) * gw for xi in x]
        ys = [max(d[key], 1e-3) for d in data]
        axB.bar(xs, ys, gw, color=col, edgecolor=SURF, linewidth=1.0, label=name)
    axB.set_yscale("log")
    axB.set_ylim(1e-3, 3e4)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=9, color=INK)
    axB.set_ylabel("mean solve time  (s, log)", fontsize=10, color=INK)
    axB.set_title("Solve time per method", fontsize=11, color=INK, pad=8)
    for s_ in ("top", "right", "left"):
        axB.spines[s_].set_visible(False)
    axB.spines["bottom"].set_color(MUTED)
    axB.tick_params(colors=MUTED, length=0)
    axB.legend(loc="upper left", frameon=False, fontsize=8.5, handlelength=1.1)

    fig.suptitle("MaxSAT (exact)  vs  GA-BP  vs  Khouzani-MILP baseline  —  four attack-graph datasets",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = os.path.join(ROOT, "comparison", "khouzani_comparison.png")
    out_pdf = os.path.join(ROOT, "comparison", "khouzani_comparison.pdf")
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=SURF)
    print("saved:", out_png)
    print("saved:", out_pdf)

    # also print the numbers behind the figure
    print("\n=== figure data ===")
    for d in data:
        print(f"{d['label'].replace(chr(10),' '):<26} n={d['n']:>3} | "
              f"Kh-vs-SAT {d['kh_v_sat']} Kh-vs-GA {d['kh_v_ga']} GA-vs-SAT {d['ga_v_sat']} | "
              f"t(s) SAT={d['t_sat']:.3f} GA={d['t_ga']:.1f} Kh={d['t_kh']:.3f}")


if __name__ == "__main__":
    main()
