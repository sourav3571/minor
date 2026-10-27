# Define and test the adaptive component

“Adaptive” has no one universal meaning. It could mean adapting distillation strength to student capacity, changing weights during training based on teacher/student disagreement, changing objective priorities per device/application, or feeding deployment measurements back into search. The problem statement specifically says teacher-guidance strength adjusts dynamically based on how small or complex the current student is. This textbook operationalizes that sentence as **candidate-conditioned KD strength**.

## Measurable rule

The student search space has depth `d` and base width `w`; use a transparent capacity proxy `q=d*w²`. Let `q_min` and `q_max` be extrema over the declared legal search space. Normalize `c=(q-q_min)/(q_max-q_min)`. Set `alpha=0.2+0.6*c`. The loss is `alpha*CE + (1-alpha)*T²*KL`; therefore the KD coefficient ranges from 0.8 (smallest proxy) to 0.2 (largest proxy). Record `q`, `c`, and alpha per candidate.

This is not a universal optimal schedule. Parameter count is only a rough complexity proxy and omits operation type; depthwise models with similar parameters can have different latency. One alternative is a lookup by measured model capacity, or a dynamic within-training schedule based on validation/disagreement—but those are different hypotheses and must be separate ablations.

## Why this is a hypothesis

Small students may need stronger guidance because limited capacity makes fitting the hard-label task difficult. Conversely, teacher targets may overwhelm a tiny or poorly matched student. Larger students may need less help. The direction and linear mapping require evidence.

## Falsifiable experiment

Train the same candidate configs and seeds under:
1. no KD (`alpha=1`),
2. fixed KD (`alpha=0.5`),
3. capacity-adaptive KD (the formula above),
4. optional reversed mapping (`alpha=0.8-0.6*c`).

Keep teacher, data, student initialization protocol, optimizer, epochs, T, and compute budget controlled. Compare validation and final test accuracy, latency, size, and candidate-level accuracy vs complexity. Report means/spread across seeds. If adaptive does not improve the predeclared outcome or costs excessive training/search overhead, report that negative result. Do not claim adaptation works solely because alpha varies.
