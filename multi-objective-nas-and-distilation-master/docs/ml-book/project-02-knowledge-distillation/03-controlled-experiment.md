# 3. Controlled teacher–student experiment

The runnable code trains three models sequentially: (A) teacher, (B) student with CE, (C) same student with CE+KD. B and C use the same small architecture and controlled training settings. The teacher is trained only on the training split; teacher soft targets are calculated online from each training minibatch.

Control what can vary: data split/order seed, preprocessing, student architecture, epochs, optimizer, learning rate, batch size, and evaluation. The teacher is intentionally different and larger. For stronger evidence, repeat student comparisons over multiple seeds and report mean/std; preserve the exact configuration and all runs.

The script records accuracy, parameter count, checkpoint bytes, and CPU/GPU inference latency. It warms the model, then times repeated batches; GPU measurement synchronizes around timers. Latency depends on local machine and thread/runtime settings. Energy is reported as `not_measured`: estimate only with an explicit power instrument and protocol on the target device.

Standard output distillation uses class logits. Feature-based distillation adds losses between internal activations and needs layer alignment; do not add this complexity before the baseline is verified.
