# Project 4 — Multi-objective edge AI

**Goal:** measure model resources responsibly and select a feasible trade-off across accuracy, latency, energy, and model size.

Read the three lessons and [edge-board deployment protocol](deploy-and-measure.md) before running [`code/multiobjective.py`](code/multiobjective.py). Exercises, answers, and a reporting template are in [`exercises.md`](exercises.md), [`solutions.md`](solutions.md), and [`project-report.md`](project-report.md). From `docs/ml-book/`:

```bash
python3 project-04-multi-objective-edge-ai/code/multiobjective.py
```

The script first plots explicitly illustrative metrics so Pareto calculations can be checked; it then benchmarks several actual small CNN candidates on the available local device and writes measured records. Energy is `null`/not measured unless a supported external measurement is added. Do not describe illustrative values as experiment results.
