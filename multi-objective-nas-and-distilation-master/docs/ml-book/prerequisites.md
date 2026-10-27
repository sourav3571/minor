# Necessary prerequisites (taught in context)

You already know programming fundamentals, so this book does not reteach Python syntax. It does teach the small amount of math and experiment design needed to reason about the final system.

## Arithmetic, functions, vectors, and matrices

A scalar is one number. A vector is an ordered list, such as three class scores `[1.2, 0.1, -0.4]`. A matrix is a grid of numbers, such as image pixels. A neural layer repeatedly performs weighted sums. For a vector input `x` and weight matrix `W`, `x @ W` is matrix multiplication: each output combines an input’s values with one column of weights. The CNN lessons build this from tiny examples before using PyTorch.

## Exponents, logarithms, and probability

Softmax uses exponentials to convert arbitrary class scores (logits) to positive values that sum to one. Cross-entropy and KL divergence use logarithms to penalize probability disagreement. You need the idea that a probability distribution allocates a total mass of 1 across classes; no probability-density theory is needed here.

## Derivatives and optimization

A derivative tells how a small input change affects an output. A gradient collects derivatives for many parameters. Backpropagation applies the chain rule through layers; gradient descent changes parameters opposite the loss gradient. The finite-difference experiment in Project 1 makes this direction observable before `loss.backward()` does it automatically.

## Data and research discipline

- **Training data** updates weights; **validation data** chooses settings and candidates; **test data** estimates the final chosen system’s performance once.
- **Leakage** is information from validation/test affecting training or decisions improperly. Repeated test-set tuning makes the test result optimistic.
- **Reproducibility** requires recording random seeds, split, software, hardware, and budget. A seed helps repeat an experiment but does not make every GPU operation deterministic.
- **Baseline** means a meaningful comparison under controlled conditions. **Ablation** removes one component to measure its contribution.
- **Variance** means results change with random initialization, data order, or hardware. Repeat promising runs and report spread, not only the best run.

These are not administrative details: without them, a claimed distillation or NAS improvement may be a data leak or an unfair comparison.

## Hardware measurement

Latency depends on device, runtime, input shape, batch size, thread settings, warm-up, and timer protocol. Parameter count and file size are not peak RAM, FLOPs are not latency, and a CPU energy estimate is not board power measurement. The edge project records the environment alongside every value and uses `not measured` instead of fabricated energy.
