# 3. Integrate resource trade-offs with NAS

For each candidate, collect validation accuracy, parameter count, serialized checkpoint bytes, and latency under the same local protocol. On the target board, additionally measure latency and energy with documented tools. Convert units consistently and state whether model size means state-dict bytes, ONNX bytes, or deployed engine bytes.

The included Pareto function transforms all objectives into “larger is better” scores (negates minimized dimensions), then checks pairwise dominance. It can first filter hard constraints. Accuracy is a validation metric while searching. The reserved test set remains for final evaluation after selecting a candidate/policy.

Example design decisions:

- Accuracy up, latency worse: both may remain Pareto; apply the actual latency constraint/use-case priorities.
- Smallest model inaccurate: it may be feasible but poor; Pareto does not magically fix accuracy requirements.
- Similar accuracy, different cost: a faster/smaller candidate may dominate or be preferred under a constraint.
- Meets parameter cap, misses latency: infeasible under latency cap; parameter count cannot override it.
- Hardware changes latency ranking: rerun benchmark/search on target; rankings are hardware-specific.

ONNX export converts a model graph for an interoperable runtime. TensorRT builds optimized NVIDIA inference engines. Quantization reduces numeric precision and can alter both accuracy and speed. These are deployment transformations, distinct from architecture search. Evaluate each transformed artifact for correctness, accuracy, latency, size, and hardware support; do not credit export gains to NAS without an ablation.
