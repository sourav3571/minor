# Project 2 solutions

1. Hard label says only which class is correct. Soft targets rank alternatives and convey teacher uncertainty/similarity, but may also convey teacher errors.
2. Class 1 has maximum logit. Logits are unrestricted scores; their sum need not be one and they can be negative.
3. Higher T generally makes distribution more uniform; numerical inspection is needed to quantify it.
4. Student parameters receive gradients. Teacher forward is inside `no_grad` and teacher parameters are frozen.
5. T=1: probabilities are `e²/(e²+1)≈0.881`, `1/(e²+1)≈0.119`. T=2: `e/(e+1)≈0.731`, `1/(e+1)≈0.269`.
6. With alpha 1, the KD multiplier is zero, so this is ordinary label-only CE; the KD term has no effect.
7. Shapes must match exactly because KL compares corresponding class entries. Raise a clear error before computing a loss.
8. Validation grid selection is legitimate if test remains untouched, but multiplies training cost and risks overfitting validation. Start with a small predeclared grid and use a modest tuning subset/budget, then retrain chosen settings.
9. Fill from JSON output; use actual measurements. Parameter count and checkpoint bytes do not equal peak memory or energy.
10. Aggregate each metric over seeds; report mean and standard deviation plus each seed where possible. A seed is a repeat, not a new dataset.
11. Teacher train loss alone is not enough. Compare teacher validation quality and evaluate student transfer; a highly overfit teacher can provide misleading targets.
12. Feature KD compares hidden representations; it adds another mechanism and potentially alignment projections. First compare CE vs output KD, then output KD vs output+feature KD with matched protocol.
