# Project 2 report template

Record teacher/student architectures, class order, data split, transforms, seeds, epochs, optimizer, T, alpha, and software/device. Include teacher validation quality and ensure it was frozen during student training.

| Model | Test accuracy | Parameters | Checkpoint bytes | Local latency ms/image | Energy |
|---|---:|---:|---:|---:|---|
| Teacher | measured | measured | measured | measured/protocol | not measured unless instrumented |
| Student CE | measured | measured | measured | measured/protocol | not measured unless instrumented |
| Student KD | measured | measured | measured | measured/protocol | not measured unless instrumented |

Include separate train/validation curves, KD/CE component behavior, latency protocol, multiple-seed mean/std for claims, and a limitations paragraph. The test column is populated only after settings are frozen. Never fill with textbook/example values.
