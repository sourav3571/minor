# 4. The complete training experiment

The script [`code/train_cnn.py`](code/train_cnn.py) is the runnable implementation. It fixes Python/NumPy/PyTorch seeds, divides the official training set deterministically into train and validation subsets, and evaluates the test set only after training. Review the source alongside this chapter.

## CNN shape walk

Input `(N,1,28,28)` → `Conv2d(1,16,3,padding=1)` `(N,16,28,28)` → ReLU same shape → max-pool `(N,16,14,14)` → `Conv2d(16,32,3,padding=1)` `(N,32,14,14)` → ReLU → pool `(N,32,7,7)` → flatten `(N,1568)` → linear `(N,10)`. Assertions catch shape drift early.

## Training loop vocabulary

- `model.train()` activates training behavior for layers such as dropout/batch normalization.
- `optimizer.zero_grad(set_to_none=True)` clears stale gradients.
- `logits = model(images)` runs the forward pass.
- `loss = criterion(logits, labels)` compares logits to integer class IDs.
- `loss.backward()` computes gradients.
- `optimizer.step()` updates weights.
- `model.eval()` switches evaluation behavior; `torch.no_grad()` avoids storing an unnecessary gradient graph.

The script reports per-epoch train loss/accuracy and validation loss/accuracy, saves the best validation checkpoint, reloads it, and evaluates the test split. It also exports a few prediction examples and writes a JSON run record. Saving/reloading tests that the artifact—not just an in-memory object—works.

## Metrics

Accuracy is correct predictions / examples. It can conceal class-specific failure. For class `c`, precision=`TP/(TP+FP)` answers “when predicted c, how often right?”; recall=`TP/(TP+FN)` answers “of true c, how many found?” The confusion matrix has true labels on rows and predictions on columns in this project’s implementation. Macro averaging computes each class metric then gives classes equal weight; it can be informative if class counts differ.

## Diagnose curves, not just final accuracy

- Training and validation both poor: underfitting, bad labels/preprocessing, too little training, or optimization issue.
- Training improves while validation stalls/degrades: overfitting or split mismatch.
- Loss is NaN: check inputs/learning rate/precision and invalid operations.
- One class dominates predictions: inspect label mapping, class balance, gradients, and logits.

Before experiments, predict direction and record it. Compare a dense baseline, number of channels, learning rate, and training subset size while changing one factor at a time. Keep seed/split/epoch budget fixed for a fair initial comparison. Repeat promising comparisons with multiple seeds; report mean and spread. A small model run demonstrates mechanics; its score is not a benchmark claim.

## Overfitting controls

More parameters can fit training examples but may generalize worse. Validation is the early warning. Data augmentation, weight decay, dropout, and early stopping are possible controls; add one at a time and evaluate on validation. The test set remains untouched until the final model/settings are fixed.
