# 1. Teacher, student, logits, and soft targets

A high-capacity model may achieve strong accuracy but cost more time, memory, energy, or storage. A smaller student has fewer resources and may lose accuracy. Distillation asks whether teacher predictions provide a useful training signal in addition to labels. It does not guarantee an improvement.

## Scores and probabilities

For class logits `[4, 1, 0]`, ordinary softmax assigns `exp(score)/sum(exp(scores))`. Approximate probabilities are `[0.936, 0.047, 0.017]`. The teacher strongly prefers class 0 and gives small probability to alternatives. If the teacher logits were `[4, 0, 0]`, the small alternatives would be even less informative.

Temperature `T>0` divides every logit before softmax: `softmax(logits/T)`. At `T=2`, `[4,1,0]/2=[2,0.5,0]`, approximately `[0.696,0.155,0.149]`. A larger temperature spreads probability mass, making class relationships easier for a student to observe; at very high temperature distributions become nearly uniform and may lose useful confidence structure.

Ground-truth labels are **hard targets** (e.g. class index `0`). Teacher probabilities are **soft targets** (a distribution over classes). With a dog image, a teacher might assign dog 0.8, cat 0.15, horse 0.05; that conveys “cat is more plausible than horse” beyond the hard dog label.

## Predict first

If `T=1`, teacher soft targets equal ordinary softmax. If `T` increases greatly, what happens to the distribution? Inspect these values in the code, do not infer that a higher T is always better.

The teacher's architecture is not the same as its parameter values. The teacher is trained first; the student has its own blueprint and parameters. For image outputs, both must produce logits of shape `(batch, number_of_classes)` in identical class order.
