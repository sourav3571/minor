# Troubleshooting field guide

Use the smallest reproducible batch first. Print shape, dtype, device, labels, logits, loss components, and gradient norms before changing architecture. Change one variable at a time and keep the failing command/config.

| Symptom | Likely cause | Diagnose | Smallest useful fix | Prevention |
|---|---|---|---|---|
| Incorrect tensor dimensions | Missing channel/batch axis or wrong NCHW/NHWC assumption | Print `shape`; inspect one sample and one batch | Use `unsqueeze(0)` only when a channel axis is actually missing; permute explicitly when needed | Assert expected input and output shapes at model boundaries |
| Incompatible tensor shapes | Spatial size/flatten count or class count does not match layer | Run a dummy forward and print each stage shape | Derive output dimensions or use adaptive pooling | Add shape assertions after blocks and test all configs |
| Incorrect label formats | CE expects class IDs `long` with shape `(N,)`, not one-hot float labels | Print label shape/dtype/range | Convert genuine class indices to `long`; correct dataset mapping | Assert `labels.dtype` and `0 <= labels < classes` |
| Loss does not decrease | Wrong labels, parameters not in optimizer, detached logits, bad scale/lr, or model too small | Overfit one tiny batch; inspect loss and gradients | Verify train loop order and lower/raise LR based on curve | Keep a tiny-batch overfit test and log config/curves |
| Model predicts one class | Label imbalance/mapping issue, failed gradients, bad initialization/optimization | Inspect class histogram, prediction histogram, logits, gradient norms | Fix mapping/weight handling; train with a verified small batch | Check data/label alignment before long runs; monitor confusion matrix |
| Training good, validation poor | Overfitting or train/validation distribution mismatch | Compare curves, splits, transforms, class counts | Reduce capacity or add one regularizer/augmentation | Use fixed stratified splits and report both curves |
| Gradients are `None` or stale | Detached computation, missing backward, missing zeroing, or frozen student | Inspect `parameter.grad` after backward; check `requires_grad` | Restore graph and call `zero_grad`, `backward`, `step` in order | Add a one-batch gradient smoke test |
| Teacher changes during KD | Teacher not frozen or left in training mode | Compare teacher parameter checksum; inspect `requires_grad`/`.training` | `teacher.eval(); teacher.requires_grad_(False)` and inference under no-grad | Assert no teacher grads and checkpoint teacher before KD |
| KD loss has wrong scale | Temperature omitted, wrong KL argument order, no `T**2`, or reduction mismatch | Print CE, raw KL, scaled KD, alpha, T separately | Student input is log-softmax; teacher target is softmax; use batchmean and `T**2` | Unit-test distributions at T=1 and check exact output shapes |
| KD overwhelms CE | Alpha/T poorly tuned or teacher inaccurate | Log component magnitudes and validation metrics | Increase alpha or tune T on validation; verify teacher quality | Predeclare small validation grid; never tune on test |
| Invalid NAS candidate | Illegal dimensions/op groups/kernel or downsampling too many times | Run config validation and dummy forward | Reject with a recorded reason before expensive training | Unit-test boundary configs and retain failure logs |
| Unfair candidate comparison | Different seed, epochs, data, optimizer, augmentation, or budget | Compare candidate protocol fields | Re-run with controlled budgets | Save complete per-candidate configuration and training updates |
| Search takes too long | Too many candidates/epochs, large teacher, slow data pipeline | Profile teacher, candidate, and evaluation wall time separately | Smoke on subset; reduce candidate count for wiring only | Set explicit compute budget and record actual cost |
| Latency is noisy | Warm-up, background load, thread changes, async GPU timing, thermal drift | Repeat timing blocks; report median/quantiles; inspect CPU load/temperature | Warm up, synchronize accelerator, control threads/load | Record hardware/runtime/power mode/input/batch/statistic |
| Pareto frontier is wrong | Objective direction reversed, equality treated as strict dominance, constraints ignored | Hand-test two/three known candidates | Normalize directions; require no-worse-all and strictly-better-one | Add tests for ties, trade-offs, domination, infeasible cases |
| Test score looks suspiciously good | Test used for tuning or data/split overlap | Audit split indices and all selection decisions | Re-establish untouched test or use a new holdout | Lock test until final frozen evaluation |
| Results do not reproduce | Unrecorded seed/config/version/split, nondeterministic kernels, changed hardware | Compare run metadata and environment | Restore exact settings; repeat seeds | Save raw logs/checkpoints and state determinism limitations |
| `torchvision` import/download fails | Incompatible torch/torchvision versions, network/cache permissions | Print both versions and full traceback | Install the matched pinned pair; retry dataset download where allowed | Use documented Python/package versions and warm the dataset cache |
| Test evaluation changes selection | Test metrics inspected before final decision | Review logs and experiment history | Make final choice using validation only, then evaluate test once | Separate tuning and final evaluation commands/reports |
| Energy reported as zero or inferred from FLOPs | No power meter or incorrect measurement boundary | Check raw instrument data and workload duration | Report `not measured` until instrumented | Specify joules/request, instrument, sampling/integration and board state |

## Debugging sequence

1. Verify one sample and one batch (`shape`, `dtype`, min/max, label range).
2. Run a forward pass and assert logits `(N, classes)` are finite.
3. Calculate loss; confirm finite scalar.
4. Backpropagate and confirm trainable parameters have finite gradients.
5. Overfit a tiny batch to validate the loop.
6. Verify evaluation mode, splits, and metrics.
7. Only then scale data, candidates, or epochs.
