# 2. Random search, candidate budgets, and leakage

Exhaustive search evaluates every combination. If there are `d` choices across `k` independent decisions, there can be `d**k` architectures; even a small space grows rapidly. Random search samples legal candidates and is a useful baseline because it is simple, reproducible, and often explores varied combinations better than an arbitrary manual list.

The included program uses one fixed train/validation split, same epoch count, optimizer, data transform, and seeds per candidate. Each candidate is a new model trained from scratch. It stores config, validation accuracy/loss, parameter count, checkpoint size, local latency, and training history. Run seed variation for serious comparisons.

Search uses validation data repeatedly, so validation performance can be overfit by a very large search. Final test remains untouched until one configuration and training protocol are selected. Retrain the winning configuration with an agreed training/validation protocol if appropriate, then evaluate once on test. Do not call the best candidate’s validation score a test score.

**Early stopping** can reduce cost, but candidates need comparable stopping rules. A tiny budget is a screening estimate, not proof of final quality. Report total candidates, epochs/updates, wall-clock search time, device, and budget.
