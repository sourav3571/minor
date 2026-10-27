# Project 4 exercises

1. Classify parameters, serialized bytes, peak RAM, MACs, latency, throughput, energy by what they measure.
2. For the illustrative table in `02-dominance-and-pareto.md`, verify every dominance comparison and frontier.
3. Modify the Pareto function for a maximized objective and two minimized objectives; test ties and strict improvement.
4. Add a latency hard constraint and demonstrate no feasible candidate behavior.
5. Predict whether depthwise convolution must be faster than regular convolution, then benchmark with same device/protocol and discuss noise.
6. Explain why `latency_ms` cannot stand in for energy joules.
7. Write a target-device protocol for Raspberry Pi or Jetson including board/runtime, power measurement, warm-up, input, repeats, and reported statistics.
8. Export/quantize as a separate ablation and describe how to attribute accuracy/resource changes correctly.
