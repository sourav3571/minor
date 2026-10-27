# Project 3 — Neural architecture search

**Goal:** define valid candidate CNNs, train/evaluate them under a fixed budget, log results, and select a configuration using validation data.

Read [`01-architecture-search-space.md`](01-architecture-search-space.md), [`02-random-search-and-fair-evaluation.md`](02-random-search-and-fair-evaluation.md), and [`03-evolutionary-search.md`](03-evolutionary-search.md). Run [`code/search.py`](code/search.py) from `docs/ml-book/`:

```bash
python3 project-03-neural-architecture-search/code/search.py --candidates 6 --epochs 2 --train-limit 6000 --test-limit 1000
```

This is a real random-search baseline, not an exhaustive or differentiable NAS method. It trains each candidate from scratch, measures validation accuracy and local latency, records every configuration, and evaluates the validation-selected winner on test only once. Increase candidate/training budgets only after verifying the small run.
