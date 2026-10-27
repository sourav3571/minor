# 2. Dominance, constraints, and Pareto frontier

Suppose we want high accuracy and low latency/size. Candidate A dominates B if A is at least as good on every objective and strictly better on at least one. Candidate B is not worth choosing when A is feasible and measurements are trusted. A candidate is Pareto-optimal if no other candidate dominates it.

Illustrative measurements only:

| Candidate | Accuracy ↑ | Latency ms ↓ | Size MB ↓ |
|---|---:|---:|---:|
| A | 0.90 | 10 | 5 |
| B | 0.88 | 8 | 4 |
| C | 0.92 | 15 | 6 |
| D | 0.85 | 12 | 7 |

B dominates D (higher accuracy, lower latency, smaller). A does not dominate B because A is more accurate but slower and larger. C does not dominate A because it is more accurate but also slower/larger. Therefore A, B, C form a trade-off frontier; D is dominated. These values teach the algorithm and are not hardware results.

A weighted score can force one scalar choice, but units must be normalized and weights express application priorities. Hard constraints are often easier to interpret: first require latency ≤ target, size ≤ target, and energy ≤ target; then select the most accurate feasible model. If no candidate is feasible, report that—do not quietly change the limits.

A four-objective frontier may be large and does not choose one winner by itself. The application owner must define constraints or a transparent tie-breaker. If energy has not been measured, do not pretend it is optimized: constrain/select only on measured objectives and mark energy as unavailable.
