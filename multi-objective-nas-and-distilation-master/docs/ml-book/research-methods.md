# Research methods for this project

## Turn an idea into a test

A hypothesis predicts a measurable difference: “under the same candidate-training budget, capacity-conditioned KD yields higher mean validation accuracy than fixed-alpha KD.” Define direction, comparison, metric, and budget before running. A vague goal such as “make NAS better” is not testable.

## Baselines and controlled comparisons

A baseline is a credible reference method: manual small student, label-only student, fixed-KD student, or random search. An ablation changes/removes one mechanism to isolate its contribution. If an experiment changes search algorithm, distillation, data augmentation, and epochs at once, its result cannot identify which component mattered.

## Repeats and variability

Different random seeds alter initialization and batch order. Repeat key runs and report each result plus mean and standard deviation. A narrow difference from one run may be noise. Multiple seeds do not replace testing on independent data or hardware.

## Search and validation overfitting

Repeatedly choosing the highest validation result can overfit validation even without backpropagating through it. Control search budgets, log all candidates, predeclare selection rules, and reserve a final test set. Never adapt a method based on repeated test results.

## Reporting negative or incomplete results

Report failed candidates, compute budget, search time, resource measurement boundary, and missing hardware measurements. Distinguish measured values from hypotheses and illustrative calculations. A negative result can still answer whether an adaptive rule helps under a defined protocol.

## Reproducible artifact checklist

Save configuration, code version/commit if available, package versions, random seeds, dataset/split indices, model checkpoint, raw candidate JSON, timing protocol, hardware identifier, plots, and exact command. The provided scripts save core JSON/checkpoints/plots; research-quality work should additionally preserve environment lock details and repeated-seed runs.
