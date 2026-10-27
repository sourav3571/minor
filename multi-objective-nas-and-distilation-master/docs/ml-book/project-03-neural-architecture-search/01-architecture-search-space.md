# 1. What exactly does NAS search?

Keep these separate:

- **Architecture:** layer types, counts, widths, kernels, skip connections.
- **Weights:** fitted numbers produced by training a fixed architecture.
- **Training hyperparameters:** optimizer, learning rate, epochs, augmentation, etc.
- **Search space:** allowed architectural choices.
- **Search algorithm:** how candidate configurations are proposed (random, evolutionary, etc.).
- **Evaluation protocol:** how each candidate receives comparable training and validation measurements.

A configuration is a machine-readable description, e.g. `{"channels": [16, 32], "kernel": 3}`. A **model factory** converts it into a PyTorch model. Validate candidates before training: legal widths, allowed kernels, output dimensions, and a resource cap. Invalid candidates should fail early with a clear reason.

The included search explores depth (one/two/three blocks), channel width, and kernel size. It deliberately uses ordinary convolutions initially so the first search remains easy to inspect. Depthwise-separable convolution may reduce multiply-accumulate work, but actual latency must be measured: operator support and memory behavior matter.

A candidate is not useful merely because it is valid. It needs trained weights and a validation result produced under a documented budget. Reusing candidate weights (weight sharing) can save time but changes the evaluation and may bias architecture ranking; this baseline trains from scratch for interpretability.
