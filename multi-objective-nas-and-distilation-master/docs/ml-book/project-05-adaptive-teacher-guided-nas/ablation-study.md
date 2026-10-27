# Ablation study

Ablations remove or alter one mechanism while holding the rest fixed. Use matched seeds/configurations where possible and evaluate all variants with the same split and budget.

| Variant | KD | Adaptive alpha | Search | Selection | What it isolates |
|---|---|---|---|---|---|
| A | No | No | Fixed/manual | Validation accuracy | Label-only student baseline |
| B | Yes | Fixed 0.5 | Fixed/manual | Validation accuracy | Output distillation effect |
| C | Yes | Capacity-conditioned | Fixed/manual | Validation accuracy | Adaptive rule effect |
| D | Yes | Fixed 0.5 | Random | Pareto | Search + multi-objective baseline |
| E | Yes | Capacity-conditioned | Evolutionary | Pareto | Proposed integrated pipeline |

Also compare random and evolutionary search at equal candidate counts; compare accuracy-only selection against Pareto selection on the same candidate records. Report which factors are entangled: E vs A changes several components, so it cannot attribute an improvement to adaptation alone. Use B vs C for that question.

Required outcome fields: test accuracy after frozen validation choice, board latency, energy per request, model artifact size, parameters, search/training time, candidate count, and failed/invalid candidates. Energy and board metrics remain unavailable until physically measured; do not synthesize them from FLOPs or CPU time.
