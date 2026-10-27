# 3. Convolution, feature maps, and CNNs

## A small operation by hand

Take a 3×3 image and a 2×2 filter:

```text
image                 filter
1 2 0                 1  0
0 1 3                -1  1
2 2 1
```

With stride 1, no padding, and one input/output channel, the top-left output is elementwise multiply-and-sum: `1*1 + 2*0 + 0*(-1) + 1*1 = 2`. Sliding one cell right gives `2*1 + 0*0 + 1*(-1) + 3*1 = 4`. Next row left: `0*1 + 1*0 + 2*(-1) + 2*1 = 0`; bottom-right: `1*1 + 3*0 + 2*(-1) + 1*1 = 0`.

```text
feature map
2 4
0 0
```

This library operation is often called convolution, though common deep-learning APIs implement cross-correlation (the filter is not flipped). The learned-filter interpretation is unchanged because filter weights are learned. The filter emphasizes a particular local pattern; do not assume it corresponds to a neat human concept.

## From hand calculation to NumPy

```python
import numpy as np
image = np.array([[1, 2, 0], [0, 1, 3], [2, 2, 1]], dtype=float)
kernel = np.array([[1, 0], [-1, 1]], dtype=float)
out = np.empty((2, 2))
for row in range(2):
    for col in range(2):
        out[row, col] = np.sum(image[row:row+2, col:col+2] * kernel)
print(out)  # [[2. 4.]
           #  [0. 0.]]
```

## Shape mechanics

For input `(N, Cin, H, W)`, output channels `Cout`, kernel `K`, stride `S`, padding `P`, dilation `D`, output spatial width is `floor((W + 2P - D*(K-1) - 1)/S + 1)`; similarly for height. With `28×28`, kernel 3, stride 1, padding 1, the output stays `28×28`. With 2×2 pooling stride 2, it becomes `14×14`.

Each output channel has filters across all input channels; parameters of a convolution are `Cout * Cin * K * K` plus typically `Cout` biases. The same filter is reused at every spatial location—weight sharing greatly reduces parameters compared with a dense layer connected independently to every pixel.

**Stride** controls step size. **Padding** adds border values (often zeros) to preserve spatial dimensions. **Pooling** summarizes local neighborhoods, commonly by maximum or average. It reduces spatial size but discards exact location details.

## PyTorch check

```python
import torch
import torch.nn.functional as F
x = torch.tensor([[[[1., 2., 0.], [0., 1., 3.], [2., 2., 1.]]]])
k = torch.tensor([[[[1., 0.], [-1., 1.]]]])
y = F.conv2d(x, k)
print(x.shape, y.shape, y)  # (1,1,3,3), (1,1,2,2), same feature map
```

A CNN chains convolutions and nonlinearities so later layers combine earlier local features into broader patterns. Pooling/stride reduce spatial resolution. A final flatten/dense classifier maps features to class logits. For CIFAR RGB input, first-layer `Cin=3`; MNIST uses `Cin=1`.
