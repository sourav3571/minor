# Project 1 solutions (reasoning, not just answers)

1. Training updates weights. Validation selects checkpoints/settings without gradient updates. Test estimates the final chosen model once.
2. `(32, 3, 64, 64)`: batch, RGB channels, height, width.
3. `2*0.5 + 1*(-1) + 0.25 = 0.25`; ReLU leaves `0.25` unchanged.
4. In the minibatch loop find `zero_grad`, `backward`, then `step`; those operations respectively clear accumulated values, calculate derivatives, and update parameters.
5. Top-left: `2*1 + 1*0 + 1*0 + 3*(-1)=-1`; top-right `1 + 0 + 3*0 + 2*(-1)=-1`; bottom-left `1 + 3*0 + 0*0 + 1*(-1)=0`; bottom-right `3 + 2*0 + 1*0 + 4*(-1)=-1`. Output `[[-1,-1],[0,-1]]`.
6. A 3×3 convolution with padding 1 keeps 28×28 before stride; stride 2 yields 14×14. First pool then yields 7×7; second conv stride 1 with padding 1 stays 7×7; second pool with kernel/stride 2 yields 3×3 for floor behavior. Flatten features become `32*3*3=288` if channel count is 32. Confirm actual module output using a dummy tensor and an assertion rather than guessing.
7. Dense parameter count depends on hidden width; calculate from each linear layer dimensions. It is usually much larger at the first layer because it connects every pixel to every hidden unit. Scores are empirical; do not invent them. The CNN's local weight sharing is the architectural reason it may be more parameter-efficient for images.
8. `0.0001` may change slowly; `0.001` often offers a useful baseline (not guaranteed); `0.1` may oscillate/diverge. These are hypotheses; use the logged loss and repeated controlled trials.
9. Use a fixed class order (0–9); a confusion matrix with true rows/predicted columns makes false negatives visible. Verify on a fabricated 2-class example before trusting dataset plots.
10. Example hypothesis: increasing channels raises capacity/parameter count and may improve validation accuracy, while raising inference time. Fix split, optimizer, epochs, seed list, and input shape; report each run plus mean/std rather than only the best.
11. Any repeated selection decision using test feedback leaks information into the model-selection process. The test set no longer approximates an untouched future sample.
12. Save `state_dict`, architecture settings, dataset transform, and seed. Recreate the model with identical architecture, load the state, set eval mode, apply the same transform, use no-grad inference, and compare tensor output shape `(1,10)` and class/probability results.
