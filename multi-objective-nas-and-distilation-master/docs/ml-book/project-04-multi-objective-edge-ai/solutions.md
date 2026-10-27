# Project 4 solutions

1. Parameters count learnable scalars; bytes measure artifact serialization; peak RAM is runtime footprint; MACs estimate arithmetic; latency is request duration; throughput is requests per time; energy is joules under a measurement boundary.
2. B dominates D. No pair dominates A, B, or C because each has a trade-off. Frontier is A/B/C.
3. Normalize direction by negating minimized values or use direction-aware comparisons. Equal objective vectors do not strictly dominate each other; one strict gain plus no losses does.
4. Filter infeasible entries before frontier selection. If none remain, return an empty feasible set and surface that condition.
5. No. Kernel implementations, memory movement, tensor sizes, and runtime support affect latency. Record repeated measurements and dispersion.
6. Power varies by state/device; joules are integral of power over time, so equal latency may draw different power.
7. Include exact board/OS/thermal/power mode, runtime and versions, model artifact, preprocessing, batch, warmups/repetitions, synchronization, power monitor/energy integration, and median/quantiles. Repeat under controlled conditions.
8. Keep original and transformed model results; run paired same-data evaluation with fixed hardware. Report conversion method and verify predictions/accuracy before claiming deployment performance.
