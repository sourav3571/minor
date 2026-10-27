# Project 1 exercises

Attempt these before opening `solutions.md`.

## Beginner
1. Label the roles of 10,000 training, 2,000 validation, and 2,000 test images. Which split may update a weight?
2. An RGB batch has 32 images of size 64×64. Write its PyTorch shape and explain every axis.
3. For inputs `(2,1)`, weights `(0.5,-1)`, bias `0.25`, calculate the neuron output before and after ReLU.
4. Read `train_cnn.py`: identify where gradients are cleared, calculated, and used.

## Intermediate
5. Manually compute a 2×2 valid convolution on `[[2,1,0],[1,3,2],[0,1,4]]` using kernel `[[1,0],[0,-1]]`.
6. Change the first layer to stride 2. Predict all subsequent spatial shapes before running; fix the classifier shape if required.
7. Add a dense-network baseline on flattened 28×28 pixels. Compare parameter counts and validation accuracy under the same data/budget.
8. Train with learning rates `0.0001`, `0.001`, and `0.1`. Predict loss behavior, run controlled trials, and distinguish instability from overfitting.

## Project-level
9. Add per-class precision/recall or a confusion-matrix plot. State class/axis conventions and test a known tiny label/prediction example.
10. Design a one-factor experiment comparing channel widths. Define hypothesis, fixed settings, budget, metrics, seeds, and reporting table before running.
11. Introduce an intentional data leakage bug by selecting a checkpoint on test accuracy in a scratch copy. Explain why the resulting test metric is invalid; do not submit that number as a result.
12. Save and reload a checkpoint in a separate process, then classify one image and show its true label, predicted label, and probabilities.
