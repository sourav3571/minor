# End-to-end experiment plan

## Hypotheses to pre-register

- H1: fixed KD changes student accuracy relative to label-only student at matched architecture/training budget.
- H2: candidate search finds a better accuracy/resource trade-off than a manually chosen student under the same total candidate-training budget.
- H3: Pareto selection retains useful trade-offs that a single accuracy objective would discard.
- H4: the capacity-conditioned adaptive alpha improves a predeclared metric over fixed alpha across multiple seeds.
- H5: search and teacher-training overhead are reported and acceptable for the research objective.

These are questions, not promised outcomes.

## Dataset and compute staging

1. Wiring smoke test: small CIFAR subset, one epoch, two candidates.
2. Development: full CIFAR-10 training data with a predeclared validation split; tune only using validation.
3. Confirmation: repeat selected comparisons over multiple seeds; optionally evaluate CIFAR-100 as a separate, harder dataset.
4. Deployment: freeze architecture/weights, export ONNX if supported, verify numerical predictions, and measure on actual named board/runtime.

Record transforms, normalization, class order, split indices/seed, versions, device, power mode, candidate budget, epochs, and wall time. CIFAR-10 and CIFAR-100 results are not interchangeable.

## Required baselines and ablations

| Question | Comparison | Hold fixed |
|---|---|---|
| Distillation value | labels vs fixed KD | same student config/data/seed/budget |
| Adaptive value | fixed KD vs adaptive KD | teacher, configs, T, optimizer, budget |
| NAS value | manual student vs searched student | total training budget, data, objective |
| Search value | random vs evolutionary | number of candidate evaluations/training budget |
| Multi-objective value | accuracy-only vs Pareto/constraints | evaluated candidates and measurements |
| Deployment value | PyTorch vs ONNX/TensorRT/quantized | same checkpoint/input/device where possible |

## Reporting tables

Candidate log: `run_id, seed, config, adaptive_alpha, train_budget, val_accuracy, val_loss, parameters, artifact_bytes, device, runtime, latency_median_ms, latency_p95_ms, energy_joules, feasible, failure_reason`. Use `not_measured`, not zero, for absent energy.

Final table: mean ± standard deviation across repeats for accuracy/resource metrics; include final test score only for frozen selection protocol. Search compute table: teacher time, candidate training time, measurement time, total wall time, and candidate count.

## Statistical caution

A seed repeat measures training randomness, not new independent data. Validation selection can overfit when many candidates are tested. Keep a final test set untouched; if possible repeat on another dataset/device. Report uncertainty and limitations rather than overclaiming tiny differences.
