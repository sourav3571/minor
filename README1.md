# Trial1.py: Experimental Overview & Limitation Analysis

This document provides a technical overview of [`Trial1.py`](file:///d:/multi-objective-nas-and-distilation-master/Trial1.py), detailing its architecture, experimental design, benchmark outcomes, and critical scientific deficiencies.

---

## 1. What `Trial1.py` Is About

`Trial1.py` is an automated, end-to-end framework combining **Multi-Objective Neural Architecture Search (NSGA-II)** with **ResNet-50 Knowledge Distillation (KD)** evaluated on the CIFAR-10 image classification benchmark. It was designed as an initial extension of **Shen et al. (*IEEE ICAIBD 2022*)**.

### Key Pipeline Components:
1. **Teacher Network (`CIFAR10ResNet50`):**
   * Adapts the standard ImageNet ResNet-50 to 32×32 images (3×3 conv1, stride 1, no initial maxpool).
   * Verified baseline: **23.52M parameters**, **2,595.66M FLOPs**, **93.55% test accuracy**.
2. **Search Space & Generative Engine:**
   * Dynamic search space based on MobileNetV2-style inverted residual blocks (`inv_res`) and depthwise separable convolutions (`depthwise`).
   * Gene bounds: Channels `[[12, 16, 20], [20, 24, 32], [32, 40, 48], [48, 64, 80]]`, Depths `[1, 2, 3]`, Kernels `[3, 5]`, Expands `[1, 2, 3]`.
3. **Multi-Objective Evolutionary Search (NSGA-II):**
   * Population: 12 candidates, 4 generations.
   * Deb's fast non-dominated sorting and crowding distance across 4 minimization objectives:
     $$\mathbf{O} = [\text{Validation Error}, \text{MFLOPs}, \text{Parameters}, \text{Energy Proxy}]$$
   * Knee-point extraction from the Rank-1 Pareto front via **TOPSIS**.
4. **Adaptive Knowledge Distillation:**
   * Capacity-ratio aware temperature annealing ($\tau \in [1.5, 5.5]$) and alpha weight ($\alpha \in [0.20, 0.85]$).
   * Includes $\tau^2$ scaling on KL divergence loss to prevent gradient starvation.
5. **Decoupled Ablation Retraining:**
   * Trains the selected student for 45 epochs with Cross-Entropy only (CE baseline).
   * Trains an identical student architecture for 45 epochs with ResNet-50 KD guidance.

---

## 2. Experimental Results Produced by `Trial1.py`

```
============================================================================================================
                            EXPERIMENTAL SCIENTIFIC BENCHMARK RESULTS (CIFAR-10)                            
============================================================================================================
Model / Framework        | Method           | Params   | FLOPs    | Lat (ms) | Energy (mJ) | Acc (%)  | F1 (%)  
------------------------------------------------------------------------------------------------------------
Shen et al. (ResNet-50)  | CE (Base Paper)  | 25.56M   | 1304.7M  | N/A      | N/A         | 95.54    | N/A
Shen et al. (Student)    | DARTS + KD       | 0.18M    | 34.88M   | N/A      | N/A         | 94.55    | N/A
------------------------------------------------------------------------------------------------------------
ResNet-50 (Teacher, Ours)| Standard CE      | 23.52M   | 2595.66M |   5.31   |     39.86   |  93.55   |  93.55
NSGA-II Student (Ours)   | No Distill (CE)  |  0.009M  |    5.15M |   1.08   |      8.12   |  76.73   |  76.70
NSGA-II Student (Ours)   | Res50-Guided KD  |  0.009M  |    5.15M |   1.08   |      8.12   |  77.01   |  77.00
============================================================================================================
```

### Verified Successes:
* **Hardware Latency Benchmarked:** Measured actual CUDA forward latency (1.08 ms, giving a **4.91x speedup** over ResNet-50), resolving a key omission in Shen et al. where latency was unmeasured (`N/A`).
* **Decoupled Distillation Setup:** Isolated the exact contribution of KD (+0.28%) on a fixed, searched student topology.

---

## 3. What `Trial1.py` Lacked (Critical Flaws & Deficiencies)

Despite executing cleanly on Colab, `Trial1.py` suffered from **five major scientific, algorithmic, and statistical limitations**:

---

### Flaw 1: Severe Under-Parameterization & Accuracy Collapse (17.5% Deficit)
* **What Happened:** The search space permitted channels as low as 12 and single-layer stages, causing the search to converge to an ultra-micro model with only **9,000 parameters (0.009M)**.
* **The Deficiency:** 
  * Shen et al.'s student achieved **94.55%** on CIFAR-10.
  * `Trial1.py`'s student achieved only **77.01%**, misclassifying nearly 1 in 4 images.
  * In computer vision literature, a 17.5% accuracy collapse is unacceptable for real-world deployment, regardless of how small or fast the model is.

---

### Flaw 2: The Teacher-Student Capacity Gap Bottleneck
* **What Happened:** The ResNet-50 teacher had 23,520,000 parameters, while the student had only 9,000 parameters—a **2,613 : 1 capacity ratio**.
* **The Deficiency:** 
  * In statistical learning theory, a 9k-parameter CNN has an extremely low Vapnik-Chervonenkis (VC) dimension. It lacks the hypothesis space necessary to emulate the complex inter-class soft probability distributions output by a 23M-parameter teacher.
  * As a result, Knowledge Distillation yielded an insignificant improvement of only **+0.28%** (76.73% $\rightarrow$ 77.01%), failing to prove that KD provided a meaningful benefit.

---

### Flaw 3: Synthetic & Collinear Energy Objective
* **What Happened:** Lines 228–230 implemented:
  ```python
  def compute_energy_proxy(latency_ms: float, p_edge_w: float = 7.5) -> float:
      return float(latency_ms * p_edge_w)
  ```
* **The Deficiency:**
  * **Collinear Double-Counting:** Because $\text{Energy} = \text{Latency} \times 7.5$, energy and latency had a correlation of **1.000**. Including both in NSGA-II and TOPSIS double-counted latency, distorting the Pareto front.
  * **Academic Red Flag:** Reporting a synthetic calculation ($L \times 7.5$) as an "Experimental Scientific Benchmark Result" without a footnote or physical hardware power instrumentation (e.g., INA219 sensor) is viewed by reviewers as fabricated experimental data.

---

### Flaw 4: TOPSIS Objective Imbalance ("3-vs-1" Distortion)
* **What Happened:** TOPSIS used weights `[Error: 0.40, MFLOPs: 0.20, Params: 0.20, Energy: 0.20]`.
* **The Deficiency:**
  * MFLOPs, Params, and Energy are all collinear efficiency metrics that shrink together.
  * Effectively, **60% of the voting weight pulled toward an empty/minimal model**, while only **40% pulled toward accuracy**.
  * TOPSIS failed to identify a balanced Pareto "knee-point"; instead, it collapsed to the degenerate minimal corner of the search space.

---

### Flaw 5: Search Space Mismatch with the Base Paper
* **What Happened:** `Trial1.py` lacked a parameter capacity constraint.
* **The Deficiency:**
  * Shen et al. specifically targeted the **0.18M (180,000 parameter)** capacity regime.
  * `Trial1.py` had no lower bound, allowing models 20x smaller than Shen et al. without maintaining task competitiveness.
  * To produce a fair, publishable comparison against Shen et al., the search space must enforce bounds near **0.08M – 0.25M parameters**.

---

## 4. Summary: Required Improvements for Successor Versions

| Metric / Dimension | `Trial1.py` (Current) | Target for Publication |
| :--- | :---: | :---: |
| **CIFAR-10 Accuracy** | 77.01% (Severe Collapse) | **$\ge 91.0\% - 93.5\%$** |
| **Student Parameter Count** | 0.009M (Under-parameterized) | **0.10M - 0.18M** (Matches Shen et al.) |
| **Distillation Gain ($\Delta \text{Acc}$)**| +0.28% (Insignificant) | **+2.0% to +3.5%** |
| **Search Objectives** | 4 (Includes duplicate Energy) | **4 Orthogonal** (`[Error, Latency, Params, FLOPs]`) |
| **TOPSIS Weight on Accuracy** | 40% (Outvoted by size metrics) | **50% Priority** ($w_{\text{error}} = 0.50$) |
| **Energy Reporting** | Claimed as measured | **Explicitly labeled as Estimated ($7.5\text{W}$ TDP)** |
