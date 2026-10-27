# 1. Edge metrics and honest measurement

Edge intelligence runs prediction near the data source (camera, phone, Raspberry Pi, Jetson). Local inference may reduce network use and latency but is constrained by CPU/GPU, memory, power, heat, storage, and runtime support.

- **Parameter count:** number of learned scalars.
- **Serialized size:** bytes in a saved artifact; depends on dtype, metadata, and format.
- **Peak memory:** runtime activation/allocator footprint; not equal to parameter count or file bytes.
- **FLOPs/MACs:** theoretical arithmetic-work counts. Libraries may count operations differently. They do not include all memory movement/launch costs and do not guarantee low latency.
- **Latency:** time for one defined request. **Throughput:** examples per second under a defined batch/concurrency.
- **Energy:** joules per request or another explicitly defined workload. It requires a measurement method; latency alone cannot establish energy.

A useful benchmark fixes model/input shape, batch size, preprocessing scope, runtime, thread count, device power mode, warm-up, repetition count, synchronization, and summary statistic (median plus spread is often useful). Benchmark after warm-up. For GPU asynchronous execution, synchronize before/after timing. Report device/software versions. CPU results are local CPU measurements only, not Raspberry Pi/Jetson results.

The script uses batch-one inference, warm-ups, repeated forward passes, and median latency. It does not time data loading or preprocessing. This protocol is not identical to an application end-to-end latency.

## FLOPs are not a speed guarantee

Depthwise convolution can reduce arithmetic but may map poorly to a particular CPU or runtime. A wider regular convolution might run faster if the library kernel is highly optimized. Measure the target hardware, not only MACs.
