# Project 5 exercise guidance

1. CIFAR benchmark → split/data loaders; teacher → trained checkpoint; student blocks → config/model factory; adaptive teacher guidance → alpha formula/log; multi-objective metrics → candidate record/selector; board deployment → exported artifact/physical timing; paper → reproducible tables/limitations.
2. `c=(q-qmin)/(qmax-qmin)`, so smallest proxy gives c=0, alpha=0.2/KD coefficient 0.8; largest gives c=1, alpha=0.8/KD coefficient 0.2.
3. It changes only the adaptation schedule while keeping distillation present in both; labels vs adaptive KD confounds distillation and adaptation.
4. Match candidate count, per-candidate epochs/updates, data/splits, seeds, training protocol, resource measurement, and validation objective. Include optimizer overhead and search wall time.
5. Use explicit null/not-measured plus instrument/protocol metadata; zero falsely claims measured no energy use.
6. E.g. adapt alpha from teacher-student disagreement during training, or adapt objective constraints to a device profile. Each needs fixed-baseline comparison, logged adaptation signal, and independent ablation.
7. Pareto archive plus one-field mutation omits NSGA-II's non-dominated sorting ranks, crowding distance, and formal parent/offspring population replacement; call it a Pareto-guided evolutionary baseline.
8. Any defensible choice should be tied to actual use-case limits and measured objectives. Name missing board/energy evidence; that honesty strengthens the scope rather than pretending it exists.
