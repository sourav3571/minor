# 1. Learning from examples and inspecting image data

## Ordinary programming vs learning

Ordinary software encodes a rule: `if temperature > 30: print("hot")`. In supervised learning, examples provide both an input and a target answer. The training procedure adjusts a parameterized function so its outputs match examples. The learned model is not a hand-written list of rules; it is a computation graph whose weights encode statistical patterns found during training.

A tiny dataset could be `x = [1, 2, 3]`, labels `[2, 4, 6]`. A model `prediction = w*x` starts with a poor `w`; training changes `w` to reduce a loss. In image classification, `x` is a pixel tensor, and the label is a class ID such as `7` for digit seven.

- **Training split:** gradients update parameters here.
- **Validation split:** compare designs/settings and decide when to stop; do not train weights on it.
- **Test split:** final held-out estimate after decisions are frozen. Do not repeatedly tune based on it.

## Image values and shapes

An MNIST grayscale image has 28 rows by 28 columns. A pixel is an intensity, commonly stored as an integer 0–255 before conversion. PyTorch batches use `(N, C, H, W)`: batch count, channels, height, width. Thus one grayscale image can have shape `(1, 28, 28)` and a batch of 64 has `(64, 1, 28, 28)`. A classifier for ten digits produces `(64, 10)` logits.

Run this inspection once Python dependencies are installed:

```python
from torchvision import datasets, transforms

images = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
image, label = images[0]
print(image.shape, image.dtype, image.min().item(), image.max().item(), label)
print(image[0, :5, :8])
```

`ToTensor()` converts grayscale pixels to floating-point values in `[0, 1]`, and gives the sample a channel dimension. The printed top-left patch is the actual pixel data, not an illustrative fabricated image. Plot with `plt.imshow(image[0], cmap="gray")`.

## Prediction vs probability

A model's raw class scores are logits. The largest logit identifies the chosen class (`argmax`). A probability distribution can be produced with softmax, but training's `CrossEntropyLoss` expects logits and internally combines log-softmax with negative log likelihood. Do not apply softmax before that loss.

## Predict before running

If a model sees shuffled labels, should validation accuracy rise consistently above chance (10 classes: about 10%)? It should not. If it does, check label/data alignment, split leakage, or metric code.
