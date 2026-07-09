"""Zenitani (2022) monotonic gradient-descent baseline, adapted to the P/E/C/D model.

Faithful reproduction of Zenitani, "A multi-objective cost-benefit optimization
algorithm for network hardening" (Int. J. Inf. Secur. 2022): the monotonic
gradient-descent search (Algorithm 1) plus the optional neighbour iterative
improvement (Algorithms 2/3).

Mapping the paper's problem onto this project's model
-----------------------------------------------------
  paper                                  ->  here
  patchset  P subseteq V (vulns removed) ->  P subseteq bn["D"] (defenses deployed)
  risk(G, V\\P)  (monotone decreasing)    ->  P_expected_loss  (deploying D forces
                                              its C=0 -> exploit probs only drop ->
                                              expected loss monotone decreases)
  cost(P)        (monotone increasing)   ->  D_cost + C_benefit_lost: the DIRECT
                                              deployment cost PLUS the business
                                              benefit forfeited by forcing every
                                              defended C=0 (an opportunity cost).
                                              Both terms are monotone increasing.
  multi-vector (rho_1..rho_n, tau)       ->  ONE risk axis (total expected loss) vs
                                              this generalized cost -> a bi-objective
  answer = Pareto set                    ->  Pareto set, then pick the single point
                                              maximising the shared scalar objective
                                              (same selection protocol as Khouzani)

Why fold C_benefit into the cost axis: the shared objective is
    objective = C_benefit_retained - P_expected_loss - D_cost.
Deploying a defence lowers P_expected_loss but ALSO lowers C_benefit_retained
(the forfeited benefit) while raising D_cost -- so C_benefit is a third, D_cost-
sized consequence of hardening. Since C_benefit_full (the no-defence benefit) is
a constant independent of D, writing cost = D_cost + (C_benefit_full -
C_benefit_retained) gives objective = C_benefit_full - (risk + cost). Hence
maximising the objective is EXACTLY minimising (risk + cost), so the gradient
path, the Pareto filter and the final selection all live on one consistent plane.
Leaving C_benefit out of the cost axis (an earlier version did) understates every
gradient's denominator by ~half and can drop the true optimum from the front.

Why the reduction is legitimate: Zenitani REQUIRES the risk function to be
monotone in the patchset. In this model deploying a defence forces C=0, which
under the noisy-OR factors can only lower every downstream exploit/privilege
probability, so P_expected_loss is monotone non-increasing while D_cost is
monotone increasing -- exactly the paper's precondition. BP is approximate so
monotonicity can wobble slightly; the paper notes Algorithm 1's local-dominance
filter (line 11) is robust to such non-monotonic cases, so we keep it.

Fairness: every patchset is turned into a D_state={D:bool} and scored with the
SAME evaluator as MaxSAT / GA / Khouzani (`run_exact_analysis` for VE or
`run_bp_analysis` for BP). The gradient/optimisation objective (risk-per-cost)
differs from the evaluation objective (scalar C_benefit - loss - cost); that
mismatch is intentional -- see memory: objective-mismatch-intentional.

The paper measures cost as the count of security-score evaluations; we report
`n_evals` (unique BP/VE calls) as the analogous scalability metric.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison.bp_core import compute_objective_bp_style


# --------------------------------------------------------------------------- #
# Hyper-parameters (all three are the ONLY knobs of Zenitani's method; the
# crossover/mutation/pop-size knobs belong to NSGA-II, not here).
# --------------------------------------------------------------------------- #
_EPS = 1e-5              # gradient's alternative denominator when Delta cost ~ 0.
                        # Paper Sect. 5.1.4 fixes epsilon = 0.00001 exactly.
_DEFAULT_N_ITER = 20    # r: Algorithm-3 refinement rounds. Paper sweeps 10/20/30
                        # (Fig.10); 20 is the mid setting -- most of the ~11-13%
                        # hypervolume gain is already captured, cost stays modest.
_DEFAULT_SAMPLE = None  # s: random-sample size in NeighborSolutions (how many
                        # front patchsets each defence is toggled against).
                        # None = use the WHOLE front (exact, O(r*nD*|front|) evals);
                        # set an int cap on large nD to bound cost at O(r*nD*s).
_DEFAULT_MAX_K = None   # cap on Algorithm-1 descent length (max #controls it will
                        # deploy). None = full descent O(nD^2). On large nD set it a
                        # little above the expected optimum (few controls): O(max_k*nD),
                        # and dominant-set Algorithm 3 grows the tail past it if needed.


def _get_evaluator(eval_mode, bp_fast=False):
    """Return the shared objective evaluator; import pgmpy-backed exact lazily.
    `bp_fast=True` binds the closed-form loopy BP (identical output, faster)."""
    if eval_mode == "exact":
        from result_analysis.exact_analysis import run_exact_analysis
        return run_exact_analysis
    if bp_fast:
        from functools import partial
        return partial(compute_objective_bp_style, fast=True)
    return compute_objective_bp_style


class _Scorer:
    """Caches the shared evaluator over patchsets (frozensets of deployed D).

    One call yields all three quantities the algorithm needs: the risk
    (P_expected_loss), the cost (D_cost) and the scalar objective used for the
    final Khouzani-style selection.
    """

    def __init__(self, bn, values_table, evaluator):
        self.bn = bn
        self.vt = values_table
        self.evaluator = evaluator
        self.D = list(bn["D"])
        self.cache = {}          # frozenset(patchset) -> eval dict
        self.n_evals = 0
        # C_benefit with NO defence deployed: the maximum, CONSTANT business
        # benefit (independent of D). Deploying a defence forces its C=0 and
        # forfeits that C's benefit, so C_benefit_full - C_benefit_retained is
        # the opportunity cost folded into the cost axis (see module docstring).
        self.c_benefit_full = self.eval(frozenset())["C_benefit"]

    def eval(self, patchset):
        key = frozenset(patchset)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        D_state = {d: (d in key) for d in self.D}
        ev = self.evaluator(self.bn, self.vt, D_state=D_state)
        self.cache[key] = ev
        self.n_evals += 1
        return ev

    # convenience accessors ------------------------------------------------
    @staticmethod
    def risk(ev):
        return ev["P_expected_loss"]

    def cost(self, ev):
        """Generalized cost = direct D_cost + forfeited C_benefit (opportunity
        cost). Monotone increasing in the patchset. With this definition the
        shared objective equals  c_benefit_full - (risk + cost), so maximising
        the objective is exactly minimising  risk + cost."""
        return ev["D_cost"] + (self.c_benefit_full - ev["C_benefit"])

    def c_benefit_lost(self, ev):
        return self.c_benefit_full - ev["C_benefit"]


def _dominant(candidates, scorer):
    """Locally dominant (non-dominated) patchsets in (risk, cost) space.

    A patchset A dominates B iff risk(A) <= risk(B) and cost(A) <= cost(B)
    (lower is better on both axes), with at least one strict. Mirrors the
    paper's "dominant solutions in Delta" (Algorithm 1, line 11).
    """
    scored = []
    for p in candidates:
        ev = scorer.eval(p)
        scored.append((p, scorer.risk(ev), scorer.cost(ev)))

    dominant = []
    for i, (pi, ri, ci) in enumerate(scored):
        dominated = False
        for j, (pj, rj, cj) in enumerate(scored):
            if i == j:
                continue
            if rj <= ri and cj <= ci and (rj < ri or cj < ci):
                dominated = True
                break
        if not dominated:
            dominant.append(pi)
    return dominant


def _gradient(new_patch, prev_patch, scorer):
    """Zenitani's gradient: decrease of residual risk per unit cost.

        gradient = (risk(P') - risk(P_prev)) / max(cost(P') - cost(P_prev), eps)

    Risk drops as the patchset grows, so this is <= 0; the STEEPEST (most
    negative) gradient is the best cost-performance step (paper eq. in Sect. 4.1,
    which selects arg max |gradient|).
    """
    ev_new = scorer.eval(new_patch)
    ev_prev = scorer.eval(prev_patch)
    d_risk = scorer.risk(ev_new) - scorer.risk(ev_prev)
    d_cost = scorer.cost(ev_new) - scorer.cost(ev_prev)
    return d_risk / max(d_cost, _EPS)


def gradient_descent_search(scorer, max_k=None):
    """Algorithm 1 (single risk axis): monotonic gradient-descent one-pass search.

    Starts from the empty patchset and adds one defence at a time, each step
    following the steepest risk-per-cost gradient. Returns the set of patchsets
    (frozensets) visited along the descent -- the approximated Pareto front.

    `max_k` caps the descent length: since optimal defences deploy few controls,
    stopping at `max_k` steps cuts the cost from O(nD^2) to O(max_k*nD) evaluations
    without losing the optimum, PROVIDED max_k >= the optimal deployment count.
    The dominant-set Algorithm 3 then grows the front's tail past max_k as needed,
    so max_k only needs to be slightly above the optimum. None = full descent.
    """
    D = scorer.D
    limit = len(D) if max_k is None else min(max_k, len(D))
    patch = frozenset()
    visited = [patch]
    while len(patch) < limit:
        remaining = [d for d in D if d not in patch]
        candidates = [patch | {d} for d in remaining]
        delta = _dominant(candidates, scorer)      # local-dominance filter
        visited.extend(delta)
        # steepest descent: arg max |gradient| == most negative gradient
        nxt = min(delta, key=lambda p: _gradient(p, patch, scorer))
        patch = nxt
    return visited


def neighbor_solutions(front, scorer, rng, sample):
    """Algorithm 2 (NeighborSolutions): for every defence c and a random sample
    of the current front patchsets, toggle c -- add it if absent (P+ direction)
    or remove it if present (P- direction). `sample` is the paper's s (how many
    front patchsets each defence is toggled against); None/>= means the whole
    front. Returns the set of neighbour patchsets.
    """
    front = list(front)
    out = set()
    for c in scorer.D:
        if sample is None or sample >= len(front):
            picked = front
        else:
            picked = rng.sample(front, sample)
        for p in picked:
            out.add(p - {c} if c in p else p | {c})
    return out


def _pareto_front(patchsets, scorer):
    """Non-dominated (Pareto-minimal) set in (risk, cost_gen), both minimised.

    O(n log n): sort by cost ascending (ties: risk ascending), sweep keeping a
    point only if its risk is strictly below the best risk seen at any lower-or-
    equal cost. Returns the list of surviving patchsets. Much faster than the
    O(n^2) `_dominant` for the large neighbour pools of Algorithm 3.
    """
    scored = []
    for p in {frozenset(x) for x in patchsets}:
        ev = scorer.eval(p)
        scored.append((scorer.cost(ev), scorer.risk(ev), p))
    scored.sort(key=lambda t: (t[0], t[1]))       # cost asc, then risk asc
    front = []
    best_risk = float("inf")
    for cost, risk, p in scored:
        if risk < best_risk - 1e-12:              # strictly better than all cheaper points
            front.append(p)
            best_risk = risk
    return front


def iterative_improvement(front, scorer, n_iter, rng, sample=None):
    """Algorithm 3 (IterativeImprovement): each round, pool the neighbours of the
    newly-added front points with the current front and RECOMPUTE the non-dominant
    set over that pool -- the "find the new dominant set among the solutions" step
    (paper Fig. 5/6). `n_iter` = paper's r, `sample` = paper's s.

    Unlike a per-member replacement, recomputing the dominant set RETAINS new
    non-dominated neighbours even when they dominate no existing member -- in
    particular the tail-extending P+ (add one more defence: lower risk, higher
    cost) is always non-dominated, so the front GROWS toward the true knee. This
    is what lets Algorithm 1 stop early (small max_k) and rely on iteration to
    reach optima with a slightly larger deployment count. Growth self-limits once
    risk hits 0 (further additions only raise cost -> dominated -> dropped).

    Follows the paper's `P intersect Q` restriction (Algorithm 3, line 5): only
    points newly added in the previous round are re-expanded (their neighbours
    weren't searched yet); already-expanded points are not re-searched. Converges
    (and breaks) when a round adds no new non-dominated point.
    """
    P = _pareto_front(front, scorer)
    newly = set(P)                                # round 1 seeds the whole front
    for _ in range(n_iter):
        seeds = [p for p in P if p in newly]      # P intersect Q
        if not seeds:
            break
        neigh = neighbor_solutions(seeds, scorer, rng, sample)
        newP = _pareto_front(set(P) | neigh, scorer)
        newly = set(newP) - set(P)                # points added this round
        P = newP
        if not newly:                             # converged: nothing new survived
            break
    return P


def find_best_defense_zenitani(bn, values_table, eval_mode="exact",
                               n_iter=_DEFAULT_N_ITER, sample=_DEFAULT_SAMPLE,
                               max_k=_DEFAULT_MAX_K, seed=42, return_all=False,
                               bp_fast=False):
    """Zenitani monotonic gradient-descent baseline over the D-defense patchset.

    Runs Algorithm 1 (one-pass gradient descent) followed by Algorithm 3
    (dominant-set neighbour iterative improvement) -- the paper's full "one-pass +
    iterative refinement" pipeline, which is the default here. Every visited
    patchset is scored by the shared BP/VE evaluator; the returned defence is the
    patchset maximising the scalar objective (same selection rule as
    Khouzani/MaxSAT/GA), scanned over EVERY evaluated patchset for robustness.

    Hyper-parameters (see module-level constants):
        n_iter (r): Algorithm-3 refinement rounds. Default 20 (paper's mid of
                    10/20/30). n_iter=0 gives the one-pass Algorithm 1 alone.
        sample (s): NeighborSolutions random-sample size. Default None = whole
                    front (exact); set an int on large nD to cap evals at O(r*nD*s).
        max_k     : Algorithm-1 descent cap (max #controls). Default None = full
                    O(nD^2) descent; on large nD set slightly above the expected
                    optimum -> O(max_k*nD), with Algorithm 3 growing the tail past
                    it. epsilon (gradient floor) fixed at 1e-5 per paper.

    Args:
        eval_mode: "exact" (VE, needs pgmpy) or "bp" (loopy BP, pure python).
        seed:      RNG seed for the neighbour sampling (reproducibility).

    Returns:
        {
          "best_D_state": {D: bool},
          "best_objective": float,
          "best_eval": <evaluator dict>,
          "pareto": [ {n_defended, risk, cost, objective, D_selected}, ... ],
          "runtime_s", "n_evals", "n_iter", "max_k", "hit_max_k",
        }
    hit_max_k is True when the chosen defence deploys exactly max_k controls,
    which signals the cap may have truncated the optimum -- rerun with a larger
    max_k for that instance.
    """
    t0 = time.time()
    evaluator = _get_evaluator(eval_mode, bp_fast=bp_fast)
    scorer = _Scorer(bn, values_table, evaluator)
    rng = random.Random(seed)

    # Algorithm 1: one-pass monotonic gradient descent (capped at max_k).
    front = gradient_descent_search(scorer, max_k=max_k)

    # Algorithm 3: dominant-set neighbour iterative improvement.
    if n_iter > 0 and scorer.D:
        front = iterative_improvement(front, scorer, n_iter, rng, sample=sample)

    # Report the (risk, cost_gen)-dominant front, but SELECT the best defence by
    # the shared scalar objective over EVERY evaluated patchset (robust to the
    # max_k cap: Algorithm-3 tail neighbours are evaluated even if not kept).
    front_set = {frozenset(p) for p in front}
    dominant = _pareto_front(list(front_set), scorer)

    best_key = None
    best_ev = None
    for p in scorer.cache:                        # scan the full evaluation cache
        ev = scorer.cache[p]
        if best_ev is None or ev["objective"] > best_ev["objective"]:
            best_ev, best_key = ev, p

    pareto = []
    for p in sorted(dominant, key=lambda s: len(s)):
        ev = scorer.eval(p)
        pareto.append({
            "n_defended": len(p),
            "risk": scorer.risk(ev),                 # P_expected_loss
            "cost_gen": scorer.cost(ev),             # D_cost + C_benefit_lost
            "c_benefit_lost": scorer.c_benefit_lost(ev),
            "objective": ev["objective"],
            "C_benefit": ev["C_benefit"],
            "P_expected_loss": ev["P_expected_loss"],
            "D_cost": ev["D_cost"],
            "D_selected": sorted(p),
        })
    pareto.sort(key=lambda r: r["cost_gen"])

    best_D_state = {d: (d in best_key) for d in bn["D"]}
    result = {
        "best_D_state": best_D_state,
        "best_objective": best_ev["objective"],
        "best_eval": best_ev,
        "pareto": pareto,
        "runtime_s": time.time() - t0,
        "n_evals": scorer.n_evals,
        "n_iter": n_iter,
        "max_k": max_k,
        "hit_max_k": (max_k is not None and len(best_key) >= max_k),
    }
    if return_all:
        result["front"] = [sorted(p) for p in front_set]
    return result


# --------------------------------------------------------------------------- #
# Self-check on a tiny structured graph: gradient descent should tie the
# exact optimum (brute force) on small instances.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from itertools import combinations

    from generate_graph.structured_graph import generate_bn_from_root_and_reverse
    from generate_graph.number_generation import generate_node_values
    from generate_graph.config import FIXED_E_PROBS

    nP, seed = 4, 7
    bn = generate_bn_from_root_and_reverse(nP=nP, seed=seed)
    values_table = generate_node_values(bn, seed=seed, fixed_e_probs=FIXED_E_PROBS)

    EVAL_MODE = "exact"
    evaluator = _get_evaluator(EVAL_MODE)

    print(f"=== Zenitani gradient descent (nP={nP}, seed={seed}, eval={EVAL_MODE}) ===")
    res = find_best_defense_zenitani(bn, values_table, eval_mode=EVAL_MODE, n_iter=0)
    print(f"{'n_def':>6}{'risk':>10}{'Dcost':>7}{'Cben_lost':>11}"
          f"{'cost_gen':>10}{'objective':>12}  defended")
    print("-" * 72)
    for r in res["pareto"]:
        print(f"{r['n_defended']:>6}{r['risk']:>10.2f}{r['D_cost']:>7.0f}"
              f"{r['c_benefit_lost']:>11.2f}{r['cost_gen']:>10.2f}"
              f"{r['objective']:>12.2f}  {r['D_selected']}")
    print(f"\nBEST Zenitani (Alg.1): "
          f"{sorted(d for d, v in res['best_D_state'].items() if v)}")
    print(f"  objective = {res['best_objective']:.4f}, "
          f"n_evals = {res['n_evals']}, runtime = {res['runtime_s']*1000:.1f} ms")

    # brute-force ground truth on this small graph
    D = list(bn["D"])
    best_bf, best_set = -float("inf"), None
    for k in range(len(D) + 1):
        for combo in combinations(D, k):
            ev = evaluator(bn, values_table,
                           D_state={d: (d in combo) for d in D})
            if ev["objective"] > best_bf:
                best_bf, best_set = ev["objective"], combo
    print(f"\n[ref] brute-force optimum: {sorted(best_set)}  "
          f"objective = {best_bf:.4f}")
    gap = best_bf - res["best_objective"]
    print(f"[ref] gap (Alg.1 vs optimum) = {gap:.6f}  "
          f"{'(TIE)' if abs(gap) < 1e-6 else ''}")

    # default pipeline: Algorithm 1 + Algorithm 3 (r=20 iterations)
    res3 = find_best_defense_zenitani(bn, values_table, eval_mode=EVAL_MODE)
    g3 = best_bf - res3["best_objective"]
    print(f"\nBEST Zenitani (Alg.1+Alg.3, r={res3['n_iter']}): objective = "
          f"{res3['best_objective']:.4f}, n_evals = {res3['n_evals']}, "
          f"gap = {g3:.6f} {'(TIE)' if abs(g3) < 1e-6 else ''}")
