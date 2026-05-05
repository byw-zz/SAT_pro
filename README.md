# SAT-Based Attack Graph Defense Optimization

This repository provides a research prototype for defense strategy optimization in network attack graphs. It supports exact optimization with MaxSAT, heuristic search with Genetic Algorithms (GA), and approximate inference with Belief Propagation (BP).

## Overview

The project implements the following workflow:

1. Generate synthetic attack graphs with privilege, exploit, condition, and defense nodes.
2. Convert attack graphs into Weighted CNF (WCNF) instances.
3. Solve the resulting MaxSAT problem with MaxHS.
4. Compare the exact MaxSAT solution against GA- and BP-based baselines.
5. Export experiment results and visualization data for further analysis.

## Project Structure

```text
SAT_pro/
├── main.py                         # Main entry point
├── batch_run_and_analyze.py        # Batch experiment runner
├── scaling_visualizer.py           # Scaling and convergence visualization
├── replot_convergence.py           # Replot convergence curves
├── mulval_graph.py                 # MulVAL attack graph utilities
├── visualize.py                    # Graph visualization utilities
│
├── generate_graph/                 # Attack graph generation
│   ├── random_graph.py             # Random graph generator
│   ├── structured_graph.py         # Structured graph generator
│   ├── numerical_generation.py     # Numerical attributes for random graphs
│   └── number_generation.py        # Numerical attributes for structured graphs
│
├── graph2sat/                      # Attack graph to WCNF conversion
│   ├── graph2sat.py                # Conversion with propagation constraints
│   ├── graph2sat_without_pro.py    # Conversion without propagation constraints
│   ├── attackgraph2sat.py          # Attack graph-specific WCNF encoding
│   └── attackgraph2sat_without_pro.py
│
├── comparison/                     # Method comparison
│   ├── scale_comparison.py         # Scale-based comparison
│   ├── batch_comparison.py         # Batch comparison runner
│   └── bp_core.py                  # BP implementation
│
├── evaluation/                     # Evaluation and visualization
│   ├── sat_ga_visualization.py     # SAT vs. GA visualization
│   └── analysis_report.txt         # Analysis report
│
├── result_analysis/                # Result analysis scripts
├── scaling_results/                # Scaling experiment outputs
├── testnet/                        # Test network graphs
└── environment.yml                 # Conda environment specification
```

## Node and Edge Types

The attack graph contains four node types:

| Type | Meaning |
|------|---------|
| `P` | Privilege nodes representing attacker capabilities or system access levels |
| `E` | Exploit nodes representing vulnerabilities or attack actions |
| `C` | Condition nodes representing system states or security conditions |
| `D` | Defense nodes representing deployable countermeasures |

The main edge types are:

- `P → E`: a privilege can be obtained through an exploit;
- `E → P`: an exploit requires a prerequisite privilege;
- `E → C`: an exploit requires a prerequisite condition;
- `C → D`: a condition is associated with a defense option.

## MaxHS Installation

This project uses **[MaxHS](https://github.com/fbacchus/MaxHS)** as the MaxSAT solver. MaxHS is included as a Git submodule and must be configured with IBM CPLEX before use.

### Prerequisites

- **IBM CPLEX Optimization Studio** — required for linking. Available free under the [IBM Academic Initiative](https://www.ibm.com/academic) for faculty and graduate students in academia.

### Setup Steps

1. **Initialize the MaxHS submodule** (from the project root):

```bash
git submodule update --init --recursive
```

2. **Configure and build MaxHS**. Follow the instructions at [https://github.com/fbacchus/MaxHS](https://github.com/fbacchus/MaxHS): set the `LINUX_CPLEXLIBDIR` and `LINUX_CPLEXINCDIR` (or `DARWIN_*` on macOS) variables in the `MaxHS/Makefile`, then run `make`.

3. The MaxHS binary will be built at `MaxHS/build/release/bin/maxhs`. Pass this path via the `--maxhs-bin` argument (or set it in code) when running the project scripts.

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate bag
```

Main dependencies include Python 3.10+, `networkx`, `numpy`, `scipy`, `matplotlib`, `pgmpy`, `torch`, `torch-geometric`, and `python-sat`.

## Usage

### Generate and solve a random attack graph

```bash
python main.py -t random --nP 10 --nE 24 --nC 30 --nD 15
```

### Generate and solve a structured attack graph

```bash
python main.py -t structured --nP 30
```

### Run batch comparison on structured graphs

From the project root directory:

```bash
python comparison/batch_comparison.py structured \
    --num-graphs 10 \
    --nP-min 151 \
    --nP-max 199 \
    --seed 42 \
    --top-n 10 \
    --save-result \
    --result-json structured_151_199_10graphs.json \
    --bp-max-iters 100 \
    --bp-damping 0.2
```

If running inside the `comparison/` directory, use:

```bash
python batch_comparison.py structured \
    --num-graphs 10 \
    --nP-min 151 \
    --nP-max 199 \
    --seed 42 \
    --top-n 10 \
    --save-result \
    --result-json structured_151_199_10graphs.json \
    --bp-max-iters 100 \
    --bp-damping 0.2
```

### Run batch comparison on random graphs

```bash
python comparison/batch_comparison.py random \
    --num-graphs 100 \
    --nP 10 \
    --nE 24 \
    --nC 30 \
    --nD 15 \
    --top-n 10 \
    --save-result \
    --result-json random_10_100graphs.json \
    --bp-max-iters 100 \
    --bp-damping 0.2
```

### Run scaling experiments

```bash
python batch_run_and_analyze.py \
    --scales 0.1 0.5 1.0 2.0 \
    --output-dir scaling_results/
```

### Visualize convergence results

```bash
python scaling_visualizer.py \
    --results-dir scaling_results/ \
    --output scaling_results/
```

### Generate comparison figures

```bash
python evaluation/sat_ga_visualization.py \
    --base comparison/batch_outputs/ \
    --output evaluation/
```

### Process MulVAL attack graph with MaxSAT

Convert MulVAL exported CSV files (vertices and arcs) to WCNF and solve with MaxHS:

```bash
python mulval_graph.py \
    --vertices testnet/VERTICES.CSV \
    --arcs testnet/ARCS.CSV \
    --output attack_graph.wcnf \
    --save-result \
    --sat pro \
    --initial-true C4,P2
```

#### `mulval_graph.py` Arguments

| Argument | Description |
|----------|-------------|
| `--arcs` | Path to ARCS.CSV file (required) |
| `--vertices` | Path to VERTICES.CSV file (required) |
| `--arcs-order` | ARCS CSV column order: `dest_src` (default) or `src_dest` |
| `--sat` | SAT conversion function: `pro` (default, with propagation constraints) or `without_pro` |
| `--initial-true` | Nodes initially true, comma-separated (default: `C4,P2`) |
| `--maxhs-bin` | Path to MaxHS executable |
| `--maxhs-timeout` | MaxHS timeout in seconds (default: 300) |
| `--no-solve` | Only generate WCNF, skip MaxHS solving |
| `--output` | WCNF output filename (default: `attack_graph.wcnf`) |
| `--save-result` | Save solving result as JSON |
| `--visualize` | Visualize attack graph before solving |

## Main Arguments

### `main.py`

| Argument | Description |
|----------|-------------|
| `-t`, `--graph-type` | Graph type: `random` or `structured` |
| `--nP` | Number of privilege nodes |
| `--nE` | Number of exploit nodes |
| `--nC` | Number of condition nodes |
| `--nD` | Number of defense nodes |
| `--seed` | Random seed |
| `--output` | Output WCNF file |
| `--maxhs-bin` | Path to the MaxHS binary |
| `--timeout` | MaxHS timeout in seconds |

### `comparison/batch_comparison.py`

| Argument | Description |
|----------|-------------|
| `random` / `structured` | Graph generation mode |
| `--num-graphs` | Number of generated graphs |
| `--nP`, `--nE`, `--nC`, `--nD` | Fixed graph size parameters for random graphs |
| `--nP-min`, `--nP-max` | Privilege-node range for structured graphs |
| `--top-n` | Number of top candidate solutions to retain |
| `--save-result` | Save batch results to JSON |
| `--result-json` | Output JSON filename |
| `--bp-max-iters` | Maximum number of BP iterations |
| `--bp-damping` | BP damping factor |

## Output Files

The project mainly produces two types of outputs.

### WCNF files

WCNF files are used as MaxSAT solver inputs:

```text
p wcnf <num_vars> <num_clauses> <top_weight>
<weight> <literal_1> <literal_2> ... 0
```

### JSON result files

Batch experiments can export JSON files containing solver status, objective values, running time, BP settings, GA results, and selected defense configurations.

## Evaluation

The evaluation scripts support:

- convergence curve plotting;
- scalability analysis;
- MaxSAT, GA, and BP comparison;
- attack graph visualization;
- batch result aggregation.

## Compared Methods

| Method | Role |
|--------|------|
| MaxSAT / MaxHS | Exact optimization baseline |
| Genetic Algorithm | Heuristic optimization baseline |
| Belief Propagation | Approximate probabilistic inference component |
| BP + GA | Hybrid approximate baseline |

## Notes

- Use MaxSAT results as the reference when evaluating solution quality.
- Random graphs are useful for stress testing because they usually contain less regular structure and more diverse dependency patterns.
- Structured graphs are useful for controlled scalability experiments.
- For reproducible experiments, always set `--seed` and save the result JSON file.

## License

This is a research prototype. Please cite the corresponding work if this repository is used in academic research.

## Acknowledgments

This project stands on the shoulders of several open-source tools and research works:

- **[MaxHS](https://github.com/fbacchus/MaxHS)** — the core MaxSAT solver used for exact defense optimization. Developed by fbacchus. Included as a Git submodule; see the [MaxHS Installation](#maxhs-installation) section for setup.

- **[CaDiCaL](http://fmv.jku.at/cadical/)** — the underlying CDCL SAT solver integrated into MaxHS, maintained by Armin Biere.

- **[MiniSat](http://minisat.se/)** — the original CDCL SAT solver used in the core MaxHS architecture, created by Niklas Eén and Niklas Sörensson.

- **[Glucose](https://www.labri.fr/perso/lsimon/glucose/)** — the improved CDCL solver used by MaxHS for its SAT engine, developed by Gilles Audemard and Laurent Simon.

- **[python-sat / pysat](https://pysathq.github.io/)** — the Python interface for reading/writing CNF and WCNF files, developed by the PySAT team at the University of Helsinki.

- **[NetworkX](https://networkx.org/)** — used for attack graph construction and manipulation.

- **[PyMoo](https://pymoo.org/)** — used for the Genetic Algorithm (NSGA-II) baseline implementation.

- **[pgmpy](http://pgmpy.org/)** — used for probabilistic modeling in the Belief Propagation component.

- **[PyTorch](https://pytorch.org/)** and **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** — used for the neural BP and GNN-based BP components.

- **MulVAL** — the network security analysis framework that provides the attack graph format used by `mulval_graph.py`. See Ou, Xinming, Wayne Fulp, and Ronald Farl. *"A Graph-Based Network Security Model"*, and Ammann, Pam, and Joel. *"Scalable, Graph-Based Network Vulnerability Analysis"*.

We are grateful to all the authors and maintainers of these projects for making their work publicly available.
