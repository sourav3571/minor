# Project 2 — Knowledge distillation

**Goal:** compare a trained teacher, a small student trained only on labels, and a matching student trained with teacher guidance.

Read [`01-teacher-logits-and-soft-targets.md`](01-teacher-logits-and-soft-targets.md), then [`02-distillation-objective.md`](02-distillation-objective.md) and [`03-controlled-experiment.md`](03-controlled-experiment.md). The runnable code is [`code/distill.py`](code/distill.py); exercises/answers are separate.

From `docs/ml-book/`:

```bash
python3 project-02-knowledge-distillation/code/distill.py --epochs 3 --limit 12000 --temperature 4 --alpha 0.5
```

The run trains a teacher, then label-only and distilled student models, with common splits, seeds, epochs, and student architecture. The teacher is frozen during student training. Logs and checkpoints are under `artifacts/project2/`. Small-budget results are instructional, not claims of state-of-the-art performance.
