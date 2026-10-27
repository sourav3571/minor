# Project 5 — Adaptive teacher-guided multi-objective NAS

**Objective from the actual brief:** search lightweight CIFAR-10/100 students, train with a larger teacher, adapt teacher-guidance strength based on student complexity, and select trade-offs across accuracy, latency, energy, and model size, ultimately validating on Raspberry Pi/Jetson and reporting a research paper.

This book’s runnable baseline uses CIFAR-10 by default, a compact CNN search space, output-logit distillation, local measured latency and checkpoint bytes, and Pareto-guided evolutionary mutation. It does **not** claim NSGA-II, energy optimization, TensorRT, or edge-board results. Energy is marked unavailable unless measured externally. See [`design-and-requirements.md`](design-and-requirements.md) for the faithful boundaries, [`adaptive-mechanism.md`](adaptive-mechanism.md) for the explicit adaptation, [`experiments.md`](experiments.md), [`ablation-study.md`](ablation-study.md), [`exercises.md`](exercises.md), [`solutions.md`](solutions.md), [`final-evaluation.md`](final-evaluation.md), and [`project-report.md`](project-report.md).

## Run a smoke-scale experiment

From `docs/ml-book/` after installing dependencies:

```bash
python3 project-05-adaptive-teacher-guided-nas/code/run_search.py --dataset cifar10 --train-limit 2000 --test-limit 500 --teacher-epochs 1 --student-epochs 1 --initial-population 2 --generations 1 --seed 53
```

This downloads CIFAR-10 and runs a deliberately tiny search; it tests wiring, not research quality. The teacher is trained first. Every student candidate is trained under the selected guidance mode and evaluated on validation. The evolutionary procedure selects candidates from a Pareto archive over validation accuracy (maximize), latency (minimize), and saved state-dict bytes (minimize). Test evaluation is performed once on the chosen candidate. Output and run configuration are under `artifacts/project5/`. Replot saved results with `python3 project-05-adaptive-teacher-guided-nas/code/plot_results.py artifacts/project5/results.json --output artifacts/project5/replotted.png`.

To compare mechanisms, execute matched-budget runs with `--guidance adaptive`, `--guidance fixed`, and `--guidance none`; keep seeds and all other settings fixed. For paper-quality results, use full data, multiple seeds, sufficient budgets, and target-board energy/latency protocols in the lessons. Never present smoke-run numbers as final findings.
