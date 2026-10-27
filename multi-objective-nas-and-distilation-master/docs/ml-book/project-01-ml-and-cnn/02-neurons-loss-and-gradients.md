# 2. Neurons, loss, gradients, and learning

## One neuron by hand

For inputs `x1=2`, `x2=1`, weights `w1=0.5`, `w2=-1`, and bias `b=0.25`, the weighted sum is `z = x1*w1 + x2*w2 + b = 2(0.5)+1(-1)+0.25 = 0.25`. A neuron transforms this value with an activation, e.g. ReLU: `relu(z)=max(0,z)=0.25`. Weights determine influence; bias shifts the threshold. A layer repeats this calculation for many outputs.

A network's **forward pass** evaluates layers in order and produces logits. An activation such as ReLU adds nonlinearity; stacking only linear transformations is still equivalent to one linear transformation and cannot express richer boundaries.

## Loss measures wrongness

For logits `[2, 0]`, softmax probability of class 0 is `exp(2)/(exp(2)+exp(0)) ≈ 0.881`; class 1 is about `0.119`. If the true label is class 0, cross-entropy is `-ln(0.881) ≈ 0.127`. If the true label is class 1, it is `-ln(0.119) ≈ 2.128`. Confident wrong answers receive a larger penalty. In a batch, the usual loss averages per-example losses.

## Derivative intuition before autograd

For `L(w)=(w-3)^2`, the derivative is `dL/dw=2(w-3)`. At `w=1`, loss is 4 and gradient is -4. Gradient descent uses `w_new = w - learning_rate*gradient`; at learning rate 0.1, `w_new=1.4`, which moves toward 3. A too-large learning rate can jump across the minimum; a very small one moves slowly.

For a network, the chain rule combines local derivatives layer by layer. **Backpropagation** is this efficient gradient calculation. A finite-difference check estimates the derivative as `(L(w+epsilon)-L(w-epsilon))/(2*epsilon)`; it is useful for debugging small functions but expensive for large models.

## What PyTorch does

`loss.backward()` calculates gradients for parameters that require gradients. Gradients accumulate, so `optimizer.zero_grad()` clears old ones before each batch. `optimizer.step()` applies an update (e.g. SGD or Adam) using those gradients. A gradient is not a parameter update by itself.

```python
import torch
w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3) ** 2
loss.backward()
print(loss.item(), w.grad.item())  # 4.0, -4.0
with torch.no_grad():
    w -= 0.1 * w.grad
print(w.item())  # 1.4
```

`torch.no_grad()` disables gradient recording for the enclosed computation. Here we use it only to mutate a parameter outside autograd's graph; later it is also used for evaluation and frozen teacher inference.

## A model is more than its architecture

Architecture is the blueprint; learned parameters are the fitted weights; training settings (optimizer, learning rate, epochs, augmentations) are hyperparameters. Changing any of these can change the result. NAS searches blueprints; ordinary training fits weights for a fixed blueprint.
