# 4. Deploy and measure on an edge board

This chapter is a field procedure, not a claim that the current machine has a Raspberry Pi or Jetson. The actual problem statement calls for physical board results. A desktop CPU run does not satisfy that requirement.

## Freeze the model and inputs

1. Choose the architecture/checkpoint using train and validation data only.
2. Save architecture configuration, state dict, class order, normalization constants, input shape, seed, and package versions.
3. Test the PyTorch checkpoint on a fixed input set and save predicted labels/logits as a reference.
4. Transfer the artifact and dataset to the named board. Record model format and file bytes (PyTorch, ONNX, or TensorRT engine are different artifacts).

## Export/runtime check

ONNX is an exchange format, not a speed guarantee. Export a model in evaluation mode using a representative dummy tensor and a declared opset. Run ONNX Runtime inference on the board if supported. Compare its logits with PyTorch on the same inputs, using a stated numerical tolerance; compare labels and validation accuracy as well. TensorRT applies NVIDIA-specific engine building and is available only on compatible NVIDIA hardware/software. Record TensorRT version, precision (FP32/FP16/INT8), calibration procedure for INT8, engine build options, and engine file size. Validate accuracy after every conversion/quantization step.

Do not compare a PyTorch CPU measurement on one machine with TensorRT on a Jetson and attribute the entire change to architecture; runtime, precision, hardware, and preprocessing also changed. Treat export/runtime/quantization as separate deployment ablations.

## Latency protocol

- Record exact board model/revision, CPU/GPU, RAM, OS, thermal state, power mode/clocks, cooling, runtime/library versions, thread counts, and input shape.
- Fix batch size (report batch 1 for per-request latency; measure a separate throughput workload if needed).
- State whether preprocessing and postprocessing are included. First report model-forward latency; then measure end-to-end application latency if deployment requires it.
- Warm up the runtime, synchronize asynchronous devices if needed, and collect enough repeated observations after the board reaches a stable thermal state.
- Report median and tail percentiles (e.g. p95), plus sample count. Avoid a single best timing.
- Keep other processes idle and repeat the measurement session; monitor thermal throttling.

## Energy protocol

Measure energy for a defined workload with an external power instrument (for example a calibrated inline meter or board power monitor). Record measurement location, sample rate, voltage/current/power units, integration interval, idle baseline, board mode, and workload count. A simple estimate is `E = sum(P_i * Δt_i)` joules over the inference interval. If subtracting idle power, report both gross and idle-subtracted energy and how the baseline was collected. Include warm-up or exclude it consistently. Repeat runs and report joules per request plus variation.

Board telemetry may be too coarse for a single short inference; measure a sustained repeated workload and divide by the exact inference count, while accounting for idle baseline and measurement overhead. Do not infer joules from latency or FLOPs. If no instrument is available, enter `not_measured` and explicitly list energy as an unmet evaluation objective.

## Result table and interpretation

| Candidate/artifact | Board/runtime | Accuracy | Median/p95 latency | Joules/request | File bytes | Peak RAM | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Fill from real run | exact identifier | measured | measured | measured or not measured | measured | measured or not measured | batch, precision, protocol |

A lower-latency artifact can still use more energy if its power draw is higher. Smaller files may still need large activation RAM. State which objectives were directly measured and which are proxies. Re-run the Pareto selector using target-board records; do not reuse desktop latency as if it were board latency.
