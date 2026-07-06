"""Shared utilities used across the pipeline (single source of truth)."""


def parse_maxhs_solution_str(maxhs_output, n_vars):
    """Parse the solution vector from MaxHS stdout/stderr.

    Handles the DIMACS ``v 1 -2 3 0`` form and the bit-string ``v 1101`` form,
    as well as a ``v`` token embedded in an ``o`` line.
    """
    assignment = {}
    for line in maxhs_output.splitlines():
        line = line.strip()
        if line.startswith("o "):
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "v" and i + 1 < len(parts):
                    data = " ".join(parts[i + 1:])
                    for tok in data.split():
                        if tok == "0":
                            continue
                        try:
                            lit = int(tok)
                            v = abs(lit)
                            if v <= n_vars:
                                assignment[v] = (lit > 0)
                        except ValueError:
                            continue
                    return assignment
        elif line.startswith("v "):
            data = line[2:].strip()
            if set(data) <= {"0", "1"} and len(data) > 0:
                for i, ch in enumerate(data, start=1):
                    if i > n_vars:
                        break
                    assignment[i] = (ch == "1")
            else:
                for tok in data.split():
                    if tok == "0":
                        continue
                    lit = int(tok)
                    v = abs(lit)
                    if v > n_vars:
                        continue
                    assignment[v] = (lit > 0)
    return assignment
