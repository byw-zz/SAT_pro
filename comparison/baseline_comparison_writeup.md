# Baseline Comparison

## Experimental Protocol

We compare the proposed MaxSAT-based method with three baselines: GA-BP,
Khouzani-MILP, and Zenitani's monotonic gradient-descent method. All methods
operate on identical attack graphs, node parameters, and defense candidates.

A defense strategy $D$ is evaluated using the common utility function

$$
U(D) = C_{\mathrm{benefit}}(D)
     - P_{\mathrm{expected\ loss}}(D)
     - D_{\mathrm{cost}}(D),
$$

where a larger value indicates a better defense strategy.

### Search and Evaluation Scope

The search objective and the final evaluation method must be distinguished:

- For small graphs, the final utility of every selected strategy is recomputed
  using exact variable elimination (VE).
- For medium graphs, the final utility is computed using Loopy Belief
  Propagation (BP), with a maximum of 100 iterations, damping factor 0.2, and
  convergence tolerance $10^{-6}$.
- GA-BP always uses BP-based utility evaluations during its genetic search. On
  small graphs, its selected strategy is subsequently reevaluated using exact
  VE.
- MaxSAT selects a strategy by solving the encoded WCNF problem, after which the
  selected strategy is evaluated using the same VE/BP evaluator as the
  baselines.
- Khouzani and Zenitani generate candidate strategies using their respective
  optimization objectives. Their candidates are finally selected and compared
  using the common utility $U(D)$.

Therefore, "exact MaxSAT" refers to exact optimization of the encoded WCNF
problem, rather than guaranteed global optimality under the external VE/BP
evaluation function.

## Datasets

Four synthetic datasets are used, comprising 220 attack graphs in total.

| Dataset | Graphs | P nodes | E nodes | C nodes | D nodes | Total nodes | Final evaluator |
|---|---:|---:|---:|---:|---:|---:|---|
| Structured-small | 100 | 12-17 | 19-42 | 10-20 | 4-7 | 45-86 | Exact VE |
| Random-small | 100 | 10 | 24 | 30 | 15 | 79 | Exact VE |
| Structured-medium | 10 | 152-198 | 445-586 | 186-240 | 62-80 | 847-1104 | Loopy BP |
| Random-medium | 10 | 100 | 240 | 300 | 150 | 790 | Loopy BP |

The configured privilege-node range of the structured-medium dataset is
$n_P \in [151,199]$; the generated instances contain 152-198 privilege nodes.

For random graphs, the maximum node out-degree is five and the P-to-E mapping
probability is 0.25. All datasets use base seed 42, with graph-specific seeds
derived from the graph identifier.

The numerical parameters are generated from the following ranges:

- privilege loss $P_{\mathrm{loss}}$: $[50,500]$;
- condition benefit $C_{\mathrm{benefit}}$: $[10,50]$;
- defense cost $D_{\mathrm{cost}}$: $[50,100]$;
- exploit probability $E_{\mathrm{prob}}$: sampled from 24 fixed values between
  0.02 and 0.95.

Structured graphs additionally apply level-dependent scaling, with
$p_{\alpha,\max}=0.8$, $cd_{\beta,\max}=0.5$, and
$cd_{\mathrm{floor}}=0.2$.

## Compared Methods

### MaxSAT

The proposed method converts the P/E/C/D attack graph into a weighted partial
MaxSAT instance with propagation constraints and solves it using MaxHS.

- MaxHS CPU limit: 120 seconds per graph;
- number of evaluated graphs: 220;
- observed MaxHS timeouts: 0.

The strategy returned by MaxHS is reevaluated using exact VE on small graphs and
Loopy BP on medium graphs.

### GA-BP

The GA-BP baseline is based on the Bayesian attack-graph risk-management method
of Poolsappasit et al. and uses NSGA-II to search over binary defense vectors.

Reference:

> N. Poolsappasit, R. Dewri, and I. Ray, "Dynamic Security Risk Management
> Using Bayesian Attack Graphs," *IEEE Transactions on Dependable and Secure
> Computing*, vol. 9, no. 1, pp. 61-74, 2012.
> DOI: [10.1109/TDSC.2011.34](https://doi.org/10.1109/TDSC.2011.34).

The GA parameters are:

| Parameter | Setting |
|---|---|
| Population size | 100 |
| Number of generations | 50 |
| Sampling | Binary random sampling |
| Crossover | Single-point crossover |
| Crossover probability | 0.8 |
| Mutation | Bit-flip mutation |
| Individual mutation probability | 1.0 |
| Per-bit mutation probability | 0.01 |
| Duplicate elimination | Enabled |
| BP maximum iterations | 100 |
| BP damping | 0.2 |
| BP tolerance | $10^{-6}$ |
| Independent runs per graph | 5 |

The seed of run $r$ on graph $g$ is

$$
\mathrm{seed}(g,r)=42+g+100000r,
\qquad r \in \{0,1,2,3,4\}.
$$

A total of $220 \times 5 = 1100$ GA runs are performed. For each graph, the
reported GA quality is the median objective value of the five independent runs.
The reported GA time is the median single-run time of the five runs; the figure
subsequently averages these per-graph median times over each dataset.

### Khouzani-MILP

The Khouzani baseline adapts the min-max probabilistic attack-graph optimization
method of Khouzani et al.

Reference:

> M. H. R. Khouzani, Z. Liu, and P. Malacaria, "Scalable Min-Max
> Multi-Objective Cyber-Security Optimisation over Probabilistic Attack Graphs,"
> *European Journal of Operational Research*, vol. 278, no. 3, pp. 894-903,
> 2019. DOI:
> [10.1016/j.ejor.2019.04.035](https://doi.org/10.1016/j.ejor.2019.04.035).

The P/E/C/D graph is reduced to a privilege-only attack graph:

- exploit and condition nodes are contracted into probabilistic
  privilege-to-privilege attack edges;
- each edge is annotated with the defenses capable of blocking it;
- a selected defense completely blocks the corresponding attack edge;
- the weakening parameter is fixed to $p_{\mathrm{ecl}}=0$.

The optimization settings are:

| Parameter | Setting |
|---|---|
| MILP solver | CBC |
| Threads | 1 |
| Solution method | Row generation |
| Defense budgets | 10 uniformly spaced budgets |
| Budget range | 0 to the total defense cost |
| Candidate selection | Maximum common utility $U(D)$ |

Khouzani's internal objective minimizes worst-path risk, whereas its final
candidate strategy is selected using the common utility $U(D)$. It is therefore
an adapted baseline rather than an identical-objective optimizer.

### Zenitani

The Zenitani baseline implements monotonic gradient descent followed by
neighborhood-based iterative improvement.

Reference:

> K. Zenitani, "A Multi-Objective Cost-Benefit Optimization Algorithm for
> Network Hardening," *International Journal of Information Security*, vol. 21,
> no. 4, pp. 813-832, 2022.
> DOI: [10.1007/s10207-022-00586-7](https://doi.org/10.1007/s10207-022-00586-7).

The common settings are:

- gradient denominator floor: $\epsilon=10^{-5}$;
- Algorithm 1: monotonic gradient descent;
- Algorithms 2/3: neighborhood-based iterative refinement;
- final strategy: the evaluated strategy with the largest common utility
  $U(D)$.

Scale-dependent settings are:

| Dataset scale | Iterations $r$ | Neighbor sample $s$ | Gradient-descent cap $max_k$ |
|---|---:|---:|---:|
| Small | 20 | Complete front | No cap |
| Structured-medium | 5 | 5 | 10 |
| Random-medium | 5 | 5 | 40 |

The medium-scale configuration is a computationally bounded adaptation. All ten
structured-medium instances and two random-medium instances reached the
recorded $max_k$ condition. Consequently, these results should not be described
as an unrestricted exhaustive execution of the original Zenitani method.

## Win/Tie/Loss Criterion

The left panel of the comparison figure reports the baseline result relative to
MaxSAT in the order

$$
\text{baseline win}/\text{tie}/\text{MaxSAT win}.
$$

A comparison is treated as a tie when

$$
\left|U_{\mathrm{baseline}}-U_{\mathrm{MaxSAT}}\right|
\leq
\max\left(0.5,0.001\left|U_{\mathrm{MaxSAT}}\right|\right).
$$

Thus, a tie denotes approximately equal evaluated utility and does not
necessarily indicate identical defense sets.

## Defense-Quality Results

| Dataset | GA-BP vs. MaxSAT | Khouzani vs. MaxSAT | Zenitani vs. MaxSAT |
|---|---:|---:|---:|
| Structured-small | 31 / 69 / 0 | 26 / 61 / 13 | 31 / 68 / 1 |
| Random-small | 8 / 92 / 0 | 4 / 25 / 71 | 8 / 69 / 23 |
| Structured-medium | 2 / 2 / 6 | 0 / 0 / 10 | 0 / 0 / 10 |
| Random-medium | 0 / 0 / 10 | 0 / 0 / 10 | 0 / 0 / 10 |

On small graphs, MaxSAT and GA-BP usually produce strategies with approximately
equal evaluated utility. GA-BP ties with MaxSAT on 69% of structured-small
instances and 92% of random-small instances. Under the shared external
evaluator, GA-BP obtains a higher value on 31% and 8% of these instances,
respectively.

The advantage of MaxSAT becomes more apparent at medium scale. On
structured-medium graphs, MaxSAT outperforms GA-BP, Khouzani, and Zenitani on
60%, 100%, and 100% of the instances, respectively. On random-medium graphs,
MaxSAT outperforms all three baselines on every instance.

These results indicate that the MaxSAT formulation provides stronger defense
quality at medium scale, while remaining competitive with GA-BP and Zenitani on
small structured instances.

## Recorded Optimization Time

The right panel reports the following mean recorded times in seconds on a
logarithmic scale.

| Dataset | MaxSAT | GA-BP median | Khouzani | Zenitani |
|---|---:|---:|---:|---:|
| Structured-small | 0.007 | 0.625 | 2.751 | 27.010 |
| Random-small | 0.012 | 9.596 | 0.275 | 4.161 |
| Structured-medium | 0.038 | 3443.079 | 0.334 | 873.107 |
| Random-medium | 10.087 | 545.140 | 136.024 | 1098.754 |

The timing scopes differ across methods:

- MaxSAT time includes WCNF conversion, MaxHS execution, and solution parsing,
  but excludes the subsequent common VE/BP reevaluation.
- GA-BP time is the median genetic-search time among five runs and excludes the
  final common reevaluation.
- Khouzani time is the accumulated MILP-solving time over the ten budgets and
  excludes part of the graph-reduction and common-evaluation overhead.
- Zenitani time is wall-clock time and includes its repeated VE/BP evaluations.

The mean Khouzani wall-clock times, which are not displayed in the PDF, are:

| Dataset | Khouzani wall-clock time |
|---|---:|
| Structured-small | 5.459 s |
| Random-small | 0.358 s |
| Structured-medium | 3.776 s |
| Random-medium | 141.467 s |

Accordingly, the right panel should be described as a comparison of the
recorded optimization times and their scaling trends, rather than a strictly
uniform end-to-end runtime benchmark.

## Main Finding

The experimental results show that the MaxSAT method is competitive with GA-BP
and Zenitani on small graphs and provides a clearer quality advantage on medium
graphs. MaxSAT outperforms Khouzani and Zenitani on every medium-scale instance
and outperforms GA-BP on 60% of structured-medium and 100% of random-medium
instances. It also has the lowest recorded optimization time in all four
datasets.

The results do not imply that MaxSAT is globally optimal under the external
VE/BP utility evaluator. Its exactness is limited to the encoded WCNF objective,
while the reported win/tie/loss results are based on the common post-hoc VE/BP
evaluation.

## Figure Caption

> **Comparison with GA-BP, Khouzani-MILP, and Zenitani baselines.** The left
> panel reports win/tie/loss percentages relative to the MaxSAT strategy under a
> shared VE/BP utility evaluator. Small-graph strategies are evaluated using
> exact variable elimination, whereas medium-graph strategies are evaluated
> using Loopy BP. GA-BP quality is represented by the median objective value
> from five independent runs per graph. The right panel reports the recorded
> optimization time on a logarithmic scale; the timing scope of each method is
> detailed in the text.
