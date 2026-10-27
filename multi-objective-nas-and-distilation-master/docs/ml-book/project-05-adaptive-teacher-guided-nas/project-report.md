# Research report / viva template

## Title and abstract
State the task, method, measured datasets/hardware, and results only after experiments. Do not claim improvement without evidence.

## Problem and motivation
Describe constraints and the exact brief objectives: accuracy, latency, energy, model size, CIFAR benchmark, deployment target.

## Related work
Summarize cited NAS, knowledge distillation, multi-objective optimization, and edge benchmark methods. Explain the gap your experiments address and cite sources read.

## Method
Define teacher, student search space, model factory, adaptive alpha formula, distillation objective, evolutionary operator, Pareto directions/constraints, candidate budget, and stopping/retraining protocol.

## Experimental protocol
Datasets/splits/transforms, seeds, software/hardware, training budget, timing, energy instrument, artifact format, and test-use policy.

## Results
Include baseline and ablation tables, Pareto plot, learning curves, candidate/search cost, hardware latency/energy, and variation across seeds. Fill from raw artifacts; use `not measured` for missing energy/board data.

## Discussion
Explain trade-offs, negative results, whether hypotheses held, limitations, hardware dependence, search overhead, and threats to validity.

## Conclusion and future work
Summarize supported claims only. Identify next steps such as CIFAR-100, true NSGA-II/crowding diversity, quantization, or additional board tests.

## Viva questions to rehearse
1. Why separate architecture and weights?
2. Derive your KD objective and explain temperature scaling.
3. What exactly adapts and what evidence tests it?
4. How is your evolutionary search different from NSGA-II?
5. Define Pareto dominance for your objective directions.
6. Why don't parameter count/FLOPs predict actual latency/energy exactly?
7. How did you prevent test leakage and ensure a fair baseline?
8. What is the measured contribution after accounting for search overhead?
9. Which results are unavailable because target hardware was not measured?
10. What experiment could disprove your main claim?
