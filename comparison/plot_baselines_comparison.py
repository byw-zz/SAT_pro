"""Four-way comparison figure: MaxSAT (exact) vs GA-BP vs Khouzani-MILP vs
Zenitani gradient-descent, on the four shared datasets.

Extends plot_khouzani_comparison.py by adding Zenitani as a third baseline
column. Every per-graph result is scored under the dataset's native evaluator
(VE for small graphs where nP+nC+nE < 100, else BP), so all four methods share
one ruler.

  Panel A. Defence quality vs. exact MaxSAT -- GA / Khouzani / Zenitani as
           100%-stacked win/tie/loss bars (baseline viewpoint).
  Panel B. Solve time per method per dataset (log scale).

Inputs (under comparison/ unless noted):
  - khouzani_out_{s,r}_{small,large}.json      (Khouzani rows)
  - *_{n}graphs_zenitani.json                  (Zenitani rows, zenitani_vs_stored.py)
  - the four stored datasets                   (GA/MaxSAT rows + summary)
  - rerun_ga/*_ga_fast.json                    (optimized GA-BP reruns)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, khouzani_out, zenitani_out, stored dataset, optimized GA rerun)
DSETS = [
    ("structured\nsmall",  "comparison/khouzani_out_s_small.json",
     "comparison/structured_12_17_100graphs_zenitani.json", "structured_12_17_100graphs.json",
     "rerun_ga/structured_12_17_100graphs_ga_fast.json"),
    ("random\nsmall",      "comparison/khouzani_out_r_small.json",
     "comparison/random_10_100graphs_zenitani.json", "random_10_100graphs.json",
     "rerun_ga/random_10_100graphs_ga_fast.json"),
    ("structured\nmedium", "comparison/khouzani_out_s_large.json",
     "comparison/structured_151_199_10graphs_zenitani.json", "structured_151_199_10graphs.json",
     "rerun_ga/structured_151_199_10graphs_ga_fast.json"),
    ("random\nmedium",     "comparison/khouzani_out_r_large.json",
     "comparison/random_100_10graphs_zenitani.json", "random_100_10graphs.json",
     "rerun_ga/random_100_10graphs_ga_fast.json"),
]

INK = "#0b0b0b"; MUTED = "#898781"; SURF = "#fcfcfb"
TEAL = "#299d8f"; YELLOW = "#e9c46a"; CORAL = "#d87659"; VIOLET = "#8a6bbf"
WIN = TEAL; TIE = YELLOW; LOSS = CORAL
C_SAT = TEAL; C_GA = CORAL; C_KH = YELLOW; C_ZEN = VIOLET
TIE_ABS_TOL = 1e-9


def _wtl(rows, akey, bkey, tol_abs=TIE_ABS_TOL):
    """baseline(akey) win/tie/loss vs bkey (higher obj = better)."""
    w = t = l = 0
    for r in rows:
        a, b = r.get(akey), r.get(bkey)
        if a is None or b is None:
            continue
        if a > b + tol_abs: w += 1
        elif a < b - tol_abs: l += 1
        else: t += 1
    return w, t, l


def load():
    out = []
    for label, khf, zenf, dsf, gaf in DSETS:
        ds = json.load(open(os.path.join(ROOT, dsf)))
        s = ds["summary"]
        ga = json.load(open(os.path.join(ROOT, gaf)))
        khp = os.path.join(ROOT, khf); zenp = os.path.join(ROOT, zenf)
        kh_rows = json.load(open(khp))["rows"] if os.path.exists(khp) else None
        zen_rows = json.load(open(zenp))["rows"] if os.path.exists(zenp) else None

        n = len(zen_rows or kh_rows or [])
        stored_by_id = {row["graph_id"]: row for row in ds["results"]}
        ga_rows = [{"ga_obj": row["new_ga_objective"],
                    "maxsat_obj": stored_by_id[row["graph_id"]]["maxsat_objective"]}
                   for row in ga["results"]]
        kh_v_sat = _wtl(kh_rows, "khouzani_obj", "maxsat_obj") if kh_rows else None
        zen_v_sat = _wtl(zen_rows, "zenitani_obj", "maxsat_obj") if zen_rows else None
        ga_v_sat = _wtl(ga_rows, "ga_obj", "maxsat_obj")

        t_sat = s.get("avg_maxsat_time_s")
        t_ga = ga["summary"].get("avg_new_ga_time_s")
        t_kh = (sum(r["khouzani_milp_ms"] for r in kh_rows) / len(kh_rows) / 1000.0
                if kh_rows else None)
        t_zen = (sum(r["zenitani_wall_ms"] for r in zen_rows) / len(zen_rows) / 1000.0
                 if zen_rows else None)
        out.append({"label": label, "n": n, "ga_v_sat": ga_v_sat,
                    "kh_v_sat": kh_v_sat, "zen_v_sat": zen_v_sat,
                    "t_sat": t_sat, "t_ga": t_ga, "t_kh": t_kh, "t_zen": t_zen})
    return out


def pct(triple):
    if triple is None:
        return None
    tot = sum(triple) or 1
    return [100.0 * x / tot for x in triple]


def main():
    data = load()
    labels = [d["label"] for d in data]
    x = list(range(len(data)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.patch.set_facecolor(SURF)

    # ---- Panel A: quality vs MaxSAT (GA / Khouzani / Zenitani) ----
    axA.set_facecolor(SURF)
    bw = 0.26
    cols = [(-bw - 0.02, "GA", "ga_v_sat"),
            (0.0, "Kh", "kh_v_sat"),
            (bw + 0.02, "Zen", "zen_v_sat")]
    for off, meth, keyname in cols:
        for i, d in enumerate(data):
            p = pct(d[keyname]); xi = x[i] + off
            if p is None:
                axA.text(xi, 50, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
            else:
                win, tie, loss = p
                axA.bar(xi, win, bw, bottom=0, color=WIN, edgecolor=SURF, linewidth=1.1)
                axA.bar(xi, tie, bw, bottom=win, color=TIE, edgecolor=SURF, linewidth=1.1)
                axA.bar(xi, loss, bw, bottom=win + tie, color=LOSS, edgecolor=SURF, linewidth=1.1)
            axA.text(xi, -4, meth, ha="center", va="top", fontsize=7.5, color=MUTED)
    axA.set_ylim(0, 100)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9, color=INK)
    axA.tick_params(axis="x", length=0, pad=22)
    axA.set_ylabel("percentage (%)", fontsize=10, color=INK)
    axA.set_title("Baseline vs. exact MaxSAT   (win / tie / loss)", fontsize=11, color=INK, pad=8)
    for s_ in ("top", "right", "left"):
        axA.spines[s_].set_visible(False)
    axA.spines["bottom"].set_color(MUTED); axA.tick_params(colors=MUTED)
    legA = [Patch(facecolor=WIN, label="baseline wins"),
            Patch(facecolor=TIE, label="tie (same defence)"),
            Patch(facecolor=LOSS, label="MaxSAT wins")]
    axA.legend(handles=legA, loc="lower center", bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=8.5, handlelength=1.1)

    # ---- Panel B: solve time per method (log scale) ----
    axB.set_facecolor(SURF)
    gw = 0.2
    series = [("MaxSAT", C_SAT, "t_sat"), ("GA-BP", C_GA, "t_ga"),
              ("Khouzani", C_KH, "t_kh"), ("Zenitani", C_ZEN, "t_zen")]
    for j, (name, col, key) in enumerate(series):
        xs = [xi + (j - 1.5) * gw for xi in x]
        ys = [max(d[key], 1e-3) if d[key] is not None else 1e-3 for d in data]
        axB.bar(xs, ys, gw, color=col, edgecolor=SURF, linewidth=1.0, label=name)
    axB.set_yscale("log")
    axB.set_ylim(1e-3, 3e4)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=9, color=INK)
    axB.set_ylabel("mean solve time  (s, log)", fontsize=10, color=INK)
    axB.set_title("Solve time per method", fontsize=11, color=INK, pad=8)
    for s_ in ("top", "right", "left"):
        axB.spines[s_].set_visible(False)
    axB.spines["bottom"].set_color(MUTED); axB.tick_params(colors=MUTED, length=0)
    axB.legend(loc="upper left", frameon=False, fontsize=8.5, handlelength=1.1)

    fig.tight_layout()
    out_png = os.path.join(ROOT, "comparison", "baselines_comparison.png")
    out_pdf = os.path.join(ROOT, "comparison", "baselines_comparison.pdf")
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=SURF)
    print("saved:", out_png)
    print("saved:", out_pdf)

    print("\n=== figure data (W/T/L = baseline win / tie / MaxSAT win) ===")
    for d in data:
        def f(t): return str(t) if t else "n/a"
        print(f"{d['label'].replace(chr(10), ' '):<20} n={d['n']:>3} | "
              f"GA {f(d['ga_v_sat'])} Kh {f(d['kh_v_sat'])} Zen {f(d['zen_v_sat'])} | "
              f"t(s) SAT={d['t_sat']} GA={d['t_ga']} Kh={d['t_kh']} Zen={d['t_zen']}")


if __name__ == "__main__":
    main()
