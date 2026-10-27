# Project 2 exercises

## Beginner
1. Explain the information difference between a hard label and a soft target.
2. Calculate which class wins for logits `[0.2, 1.7, -0.5]`. Why are these not probabilities?
3. Predict how `softmax([4,1,0]/T)` changes as T goes from 1 to 4.
4. In the code, identify which parameters receive gradients during student KD training.

## Intermediate
5. Implement a two-class softmax manually using exponentials for logits `[2,0]`, then repeat with T=2.
6. Set alpha to 1. Is the training objective still distillation? Explain.
7. Add a check that teacher/student logits have identical shape and class count.
8. Compare T values 1, 2, 4 and alpha values 0.25, 0.5, 0.75 on validation data using a fixed budget. Explain why this grid costs more compute and how to reduce it.

## Project-level
9. Create a table comparing teacher, label student, KD student by accuracy, parameter count, serialized bytes, and measured latency. State device and protocol.
10. Repeat student comparisons over at least three seeds; report mean and standard deviation. Do not report only the best seed.
11. The teacher reaches lower training loss but worse validation accuracy than a smaller network. Decide whether to distill from it and justify with an experiment.
12. Explain why feature distillation is a separate method and propose how to ablate it without confusing its effect with output KD.
