# Final evaluation checklist

- [ ] Dataset and class order are documented; split indices/seed are saved.
- [ ] Teacher is selected/trained using train/validation only; checkpoint metadata is retained.
- [ ] Search space, validity checks, mutation, objective directions, constraints, and budget are specified.
- [ ] Every candidate result includes config, seed, training budget, validation metrics, latency protocol, artifact size, and failure status.
- [ ] Adaptive alpha formula and per-candidate values are logged.
- [ ] Label-only, fixed-KD, and adaptive-KD baselines use comparable protocols.
- [ ] Random and evolutionary search budgets are matched.
- [ ] Pareto dominance tests include objective ties and constraints.
- [ ] Final candidate is selected without test data; test is evaluated once after freezing.
- [ ] Latency includes device/runtime/input/batch/warm-up/repeats/statistic and synchronization details.
- [ ] Energy is measured with a stated physical instrument/protocol or reported `not measured`.
- [ ] ONNX/TensorRT artifacts are validated separately; conversion/deployment effects are ablated.
- [ ] Multiple seeds, variability, compute overhead, and limitations are reported.
- [ ] Plots and tables are generated from saved raw JSON/CSV records.
- [ ] No number is described as a Raspberry Pi/Jetson result unless that hardware produced it.
