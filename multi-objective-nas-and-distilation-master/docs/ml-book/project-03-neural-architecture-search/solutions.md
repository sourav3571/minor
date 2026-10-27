# Project 3 solutions

1. Blueprint is architecture; fitted values are weights; optimizer/lr are training hyperparameters; allowed configs are search space; sampler/evolution is algorithm; data split/training schedule/metric define evaluation.
2. Test legal depth, width choices, kernel choices, and final class output. A validation function should reject malformed configs before constructing expensive modules.
3. A conv weight count is `Cin*Cout*K²` (+ biases). Doubling both Cin and Cout makes that layer's weight count about four times larger; first conv only doubles output widths if input channels remain fixed.
4. More candidates increases compute and usually improves the maximum observed score by selection chance; do not conclude every candidate improves. Record actual run results.
5. A shared stopping rule makes outcomes easier to compare; report each candidate's epochs and total training time. Validation-driven stopping uses validation, not test.
6. Mean rewards expected average; median resists outliers; lower confidence bound can favor stability. Choose before inspecting held-out test and report seed variation.
7. Every test-guided change indirectly adapts the system to the test set, making it no longer an untouched estimate.
8. Match total candidate runs and epochs/updates per candidate, dataset split, seed set, and metric. Include optimizer overhead and total wall-clock time; compare validation-selected configurations, then test them once.
