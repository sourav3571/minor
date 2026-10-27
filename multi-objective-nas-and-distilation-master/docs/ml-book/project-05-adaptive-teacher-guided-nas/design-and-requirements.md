# Requirements translation and research design

## Concrete problem
Given labeled CIFAR images and a constrained target device, find student CNN architectures with useful validation accuracy while keeping inference latency, energy per request, and deployed model size within acceptable bounds. The brief names CIFAR-10 and CIFAR-100, a high-accuracy teacher (e.g. ResNet/EfficientNet), lightweight blocks such as depthwise convolutions, evolutionary search (e.g. NSGA-II), adaptive distillation, ONNX/TensorRT deployment, physical Raspberry Pi/Jetson measurements, and a paper.

## Baseline and proposed system

- **Baseline A:** manually chosen small CNN trained on labels only.
- **Baseline B:** same student trained using fixed output distillation.
- **Baseline C:** random architecture search with fixed KD and multi-objective selection.
- **Proposed system:** evolutionary mutation of student configurations; each candidate uses an explicitly complexity-conditioned KD weight; candidate records contain validation accuracy, target-runtime latency, energy, and model artifact size; a constrained Pareto decision chooses a deployable student.

The supplied runnable implementation covers CIFAR-10, a CNN teacher trained from scratch, regular/depthwise-separable student choices, CE/KD training, an evolutionary Pareto-archive baseline, local latency, parameter/checkpoint size, validation selection, and one final test. It is intentionally a research scaffold, not yet the full hardware system.

## Teacher and student

The teaching implementation uses a CNN with three stages for the teacher and one-to-three configurable student blocks. It is CPU-capable and avoids downloading a pretrained model. This differs from the brief’s suggested ResNet/EfficientNet. A research extension can replace it with a CIFAR-adapted ResNet and document initialization, licensing, preprocessing, and compute. Teacher and student output indices must share identical CIFAR class order.

## Search space and objective

Configuration: depth ∈ {1,2,3}, width ∈ {8,16,24,32}, operation ∈ {regular, depthwise-separable}, kernel ∈ {3,5}. Invalid dimensions or runtime failures should be logged, not silently dropped. A trained candidate yields validation accuracy (maximize), batch-1 latency on the declared device/runtime (minimize), serialized student weight bytes (minimize); parameter count is reported as an additional proxy. Energy is a fourth required objective only when measured with a declared power meter and boundary. This local implementation cannot truthfully optimize energy.

## Adaptive definition (testable, not universal)

Let `c` be the candidate’s normalized complexity proxy based on depth and width squared, scaled to `[0,1]` across the legal search space. Set CE weight `alpha = 0.2 + 0.6*c`; KD weight is `1-alpha`. Thus more constrained/smaller candidates receive relatively more teacher guidance. The mapping is a hypothesis, not a law. Compare against constant alpha (fixed KD), alpha=1 (label-only), and optionally reversed adaptation. Record alpha for every candidate.

## Search and selection strategy

The code uses Pareto dominance to preserve non-dominated candidates, then mutates one configuration field from a Pareto parent. This is an understandable evolutionary baseline, not a claim to implement NSGA-II: it lacks NSGA-II crowding-distance selection and formal population replacement. A paper implementation should either call a validated NSGA-II library and document its operators, or implement/test non-dominated sorting plus crowding distance.

Only training data updates parameters. Validation evaluates candidates and chooses settings. The test split is reserved until one final candidate is frozen. Use multiple seeds and report distributions. If hard resource limits are available, filter by latency/size/energy and report no-feasible-candidate outcomes; otherwise show the Pareto set and let deployment priorities choose transparently.

## Minimum credible contribution

The brief’s strongest proof requires (1) matched label-only/fixed-KD/adaptive-KD ablation; (2) random vs evolutionary search at equal candidate-training budget; (3) single- vs multi-objective selection; (4) adaptive vs constant KD strength; (5) actual target-board latency and energy; and (6) compute/search overhead. Without measured physical-device power and performance, describe the work as a simulation/local benchmark, not validated edge deployment.
