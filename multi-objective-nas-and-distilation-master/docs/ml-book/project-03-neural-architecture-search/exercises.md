# Project 3 exercises

1. Label blueprint, fitted weights, training hyperparameter, search space, search algorithm, evaluation rule in a described run.
2. Add an architecture validity function and tests for a valid and invalid configuration.
3. Predict how parameter count changes when both channel widths double. Derive convolution parameter scaling.
4. Run random search with 2, 4, 8 candidates; compare best validation score and search cost. Explain why best-so-far tends to improve with more trials even if candidates are not better on average.
5. Implement early stopping on validation loss. Define comparable patience/budget and document its effect on total compute.
6. Add candidate-level seed repetition; decide whether to rank on mean, median, or a conservative statistic, and justify.
7. Explain how repeatedly searching against the test set leaks information even though no gradients use test examples.
8. Propose a fair budget-matched random vs evolutionary comparison and specify what counts as one evaluation.
