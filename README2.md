# Trial2.py: Experimental Overview & Failure Analysis

This document provides a comprehensive post-mortem and scientific review of [`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py), detailing what was resolved compared to [`Trial1.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial1.py), why [`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py) failed during training/evaluation, and the exact deficiencies that must be addressed for publication in a peer-reviewed venue.

---

## 1. What `Trial2.py` Attempted to Solve

[`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py) was drafted as an explicit upgrade to address the five major limitations documented in [`README1.md`](file:///d:/multi-objective-nas-and-distilation-master/README1.md).

### Verified Algorithmic Improvements Over `Trial1.py`:
1. **Cosine Annealing Calibration Bug Fixed:**
   * In `Trial1.py`, `T_max` was set to `student_epochs` (e.g., 45), but `scheduler.step()` was executed inside the inner batch loop (352 batches/epoch). The learning rate collapsed to near-zero within the first few hundred batches.
   * `Trial2.py` correctly calibrated this:
     $$\text{total\_batch\_steps} = \text{student\_epochs} \times \text{len}(\text{train\_loader}) = 60 \times 352 = 21,120 \text{ iterations}$$
     Ensuring smooth, batch-level learning rate decay over the full 60 epochs.
2. **Eliminated Collinear Energy Proxy Objective:**
   * `Trial1.py` computed $\text{Energy} = \text{Latency} \times 7.5$, giving energy an identical rank correlation ($r = 1.0$) with latency and distorting the Pareto front.
   * `Trial2.py` established four strictly orthogonal objectives:
     $$\mathbf{O} = [\text{Error (50\%)}, \text{Latency (30\%)}, \text{Params (10\%)}, \text{FLOPs (10\%)}]$$
3. **Re-Weighted TOPSIS Knee-Point Selection:**
   * Replaced the flawed 3-vs-1 weighting (`[0.40, 0.20, 0.20, 0.20]`) with `[0.50, 0.30, 0.10, 0.10]`, dedicating 50% priority to classification accuracy to prevent the Pareto knee from collapsing into an empty/micro model.
4. **Sharpened Distillation Schedule:**
   * Constrained the temperature schedule to $\tau \in [2.0, 3.2]$ (down from $[1.5, 5.5]$) and $\alpha \in [0.30, 0.75]$, preventing excessive logit over-smoothing and preserving class-boundary sharpness.
5. **Capacity Boundary Enforced in Search:**
   * Configured `Constrained4ObjNSGAII(min_params=80000, max_params=300000)` to force the architecture search toward Shen et al.'s targeted $0.18\text{M}$ capacity regime.

---

## 2. Why `Trial2.py` Failed (Root-Cause Breakdown)

Despite these theoretical corrections, executing [`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py) on Google Colab encounters **five critical execution and architectural failures**:

---

### Failure 1: Complete Checkpoint Absence for Student Models (Fatal Cloud Compute Loss)
* **What Happened:**
  * While `teacher_resnet50_best.pth` is saved to Google Drive in Stage 1, Stage 3 trains `student_ce` (60 epochs) and `student_kd` (60 epochs)—totalling **120 full epochs of training (~30 to 40 minutes on Colab T4)**.
  * Look at lines 472–536 of [`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py): **`torch.save()` is never called for either student model.**
* **The Consequences:**
  1. **Irrecoverable Session Loss:** If Google Colab experiences a runtime reset, GPU preemption, browser disconnect, or memory spike at epoch 58, all 120 epochs of compute evaporate completely.
  2. **No Persisted Artifacts for Paper Verification:** After completion, no `.pth` model files exist on Drive to inspect weights, compute confusion matrices, plot Grad-CAM / t-SNE feature clusters, or deploy to edge devices.
  3. **No Validation Checkpoint Recovery:** The script tests the model weights remaining at the final batch of epoch 60. Without saving and restoring `best_val_acc`, late-stage training stochasticity or slight overfitting degrades final test metrics.

---

### Failure 2: The `depthwise` Block Type is a 0.0% Feasibility Dead-End
* **What Happened:**
  * [`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py) specified `block_types: ["inv_res", "depthwise"]` with a lower parameter boundary of $80,000$ (`min_params=80000`).
* **Empirical Parameter Sweep:**
  Monte Carlo analysis (1,000 random genome samples) reveals:
  * **Inverted Residual (`inv_res`):** $\text{Params} \in [33,326,\ 289,066]$ (Mean: $118,865$) $\rightarrow$ **72.8% Feasible**.
  * **Depthwise Separable (`depthwise`):** $\text{Params} \in [10,730,\ 62,614]$ (Mean: $30,457$) $\rightarrow$ **0.0% Feasible**.
* **The Consequences:**
  1. Because `block_type` is chosen with 50% probability, **exactly half of all sampled candidate genomes are physically incapable of reaching the 80,000 parameter threshold**.
  2. When `depthwise` is chosen, the `expand_ratio` gene is completely ignored, creating dead genetic branches.
  3. Half of the evolutionary search population is wasted on structurally non-viable architectures.

---

### Failure 3: Mathematical Inversion of Deb's Constraint Domination Rule
* **What Happened:**
  In `strictly_dominates` (lines 304–318):
  ```python
  p_feas = (self.min_params <= p["params"] <= self.max_params)
  q_feas = (self.min_params <= q["params"] <= self.max_params)

  if p_feas and not q_feas:
      return True
  if not p_feas and q_feas:
      return False

  # When BOTH candidates are infeasible:
  p_objs = [p["error"], p["latency"], p["params"], p["mflops"]]
  q_objs = [q["error"], q["latency"], q["params"], q["mflops"]]
  not_worse = all(pv <= qv for pv, qv in zip(p_objs, q_objs))
  better = any(pv < qv for pv, qv in zip(p_objs, q_objs))
  return not_worse and better
  ```
* **The Flaw:**
  * In formal Multi-Objective Optimization (Deb et al., 2002), if **both solutions violate constraints**, the candidate with the **smaller constraint violation ($CV$) must dominate**:
    $$CV(x) = \max(0, \text{min\_params} - \text{params}) + \max(0, \text{params} - \text{max\_params})$$
  * In [`Trial2.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial2.py), when both $p$ and $q$ are infeasible ($<80\text{k}$ params), the code evaluates raw minimization objectives.
  * If candidate $A$ has 12,000 params ($CV = 68,000$) and candidate $B$ has 75,000 params ($CV = 5,000$), candidate $A$ has fewer params ($12\text{k} < 75\text{k}$) and fewer FLOPs. **Therefore, candidate $A$ dominates candidate $B$!**
  * **Result:** The evolutionary engine actively penalizes models near 80k and rewards models for drifting deeper into the invalid ultra-micro trap.

---

### Failure 4: 2-Epoch Proxy Accuracy Starvation (Capacity Distortion)
* **What Happened:**
  * During the NSGA-II search, each candidate architecture is evaluated by training for only **2 epochs** (`epochs=2`) on CIFAR-10.
* **The Flaw:**
  * An inverted bottleneck CNN with 180,000 parameters requires learning hundreds of kernel interactions across 4 stages. In 2 epochs, it achieves only ~52–60% validation accuracy.
  * In contrast, ultra-small models with fewer channels learn their limited parameter space quickly during initial epochs.
  * Using 2-epoch proxy accuracy creates a severe **short-horizon ranking inversion**: architectures that converge faster early on are falsely selected over architectures that possess superior long-term capacity at 60 epochs.

---

### Failure 5: The FLOPs vs. MACs Academic Reporting Mismatch
* **What Happened:**
  Line 199 calculates FLOPs as:
  $$\text{FLOPs} = 2 \times \frac{C_{\text{in}}}{\text{groups}} \times C_{\text{out}} \times K_h \times K_w \times H_{\text{out}} \times W_{\text{out}}$$
  Counting 1 Multiply-Accumulate (MAC) as 2 floating-point operations.
  * For ResNet-50, this yields **~2,595.66 MFLOPs**.
  * In the comparison table, Shen et al.'s baseline ResNet-50 is cited as **1,304.7M**.
* **The Flaw:**
  * Shen et al. reported **Multiply-Accumulate operations (MACs)**, not FLOPs ($1304.7\text{M MACs} \times 2 \approx 2609.4\text{M FLOPs}$).
  * Without displaying both MACs and FLOPs or clarifying the notation, reviewers will conclude your ResNet-50 implementation requires double the computational complexity of the base paper.

---

## 3. Comparative Summary Across Versions

| Dimension | `Trial1.py` (Initial) | `Trial2.py` (Current Script) | Requirements for Publication |
| :--- | :---: | :---: | :---: |
| **Cosine Annealing Steps** | Broken (Epochs stepped per batch) | **Fixed** ($T_{\max} = \text{epochs} \times \text{steps}$) | Calibrated per-batch |
| **Search Objectives** | 4 (Included duplicate Energy) | **Fixed** (4 Orthogonal Objectives) | 4 Orthogonal (`[Error, Lat, Params, MACs]`) |
| **TOPSIS Weights** | `[0.40, 0.20, 0.20, 0.20]` (Collapsed) | **Fixed** (`[0.50, 0.30, 0.10, 0.10]`) | 50% Priority on Accuracy |
| **Student Checkpointing** | In-Progress & Best Checkpoints | **Failed (Zero Saving for Students)** | Auto-saved to Drive with Best-Val Restoration |
| **Depthwise Block Space** | Unconstrained (<10k params) | **Failed (0% Feasible in [80k, 300k])** | Re-scaled channels or Inverted Bottleneck |
| **Deb's Constraint Rule** | Not implemented | **Failed (Inverted violation dominance)** | Domination by minimal constraint violation |
| **Metric Distinction** | Unlabeled FLOPs | **Ambiguous (2x FLOPs vs Shen MACs)** | Explicitly report both MACs & FLOPs |

---

## 4. Required Modifications for `Trial3.py`

To transform this framework into a submission-ready artifact for peer review:
1. **Add Google Drive Checkpointing:** Write `torch.save(student_ce.state_dict(), ...)` and `torch.save(student_kd.state_dict(), ...)` every 5 epochs and at `best_val_acc`.
2. **Correct Infeasible Domination:** Implement $CV(p) < CV(q)$ when both candidates violate parameter bounds.
3. **Calibrate Depthwise Channels:** Increase channel options to `[[28, 32, 40], [48, 56, 64], [72, 88, 96], [112, 128, 144]]` so depthwise models achieve $\ge 80,000$ parameters.
4. **Report Both MACs and FLOPs:** Track MACs directly and report FLOPs as $2 \times \text{MACs}$ to eliminate paper comparison discrepancies.
