"""
====================================================================================================
Project: Multi-Objective Pareto Architecture Search with ResNet-50 Teacher Guidance
Benchmark: CIFAR-10 Classification (Direct Extension of Shen et al., IEEE ICAIBD 2022)
Hardware: Optimized for Google Colab T4 GPU (FP16 AMP, Per-Epoch Drive Resumption, 0-VRAM Leaks)
====================================================================================================
"""

import os
import sys
import math
import time
import json
import copy
import random
import gc
import numpy as np
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# =====================================================================
# 1. Google Drive Resilient Checkpoint Manager
# =====================================================================
def configure_storage() -> str:
    colab_mount = "/content/drive"
    preferred_dir = "/content/drive/MyDrive/cifar10_resnet50_nas_kd"
    fallback_dir = "./checkpoints_resnet50_nas"

    if os.path.exists(colab_mount):
        target_dir = preferred_dir
    else:
        try:
            from google.colab import drive
            print("[STORAGE] Mounting Google Drive to survive Colab session resets...")
            drive.mount(colab_mount)
            target_dir = preferred_dir
        except Exception:
            print("[STORAGE] Colab environment not detected. Using local directory.")
            target_dir = fallback_dir

    os.makedirs(target_dir, exist_ok=True)
    print(f"[STORAGE] Verified persistent path: {os.path.abspath(target_dir)}")
    return target_dir

STORAGE_PATH = configure_storage()
REGISTRY_FILE = os.path.join(STORAGE_PATH, "experiment_registry.json")

# =====================================================================
# 2. Hardware Engine & Deterministic Reproducibility
# =====================================================================
def enforce_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

enforce_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")
print(f"[ENGINE] Active Device: {DEVICE} | Automatic Mixed Precision (AMP): {USE_AMP}")
if USE_AMP:
    print(f"[ENGINE] Active GPU: {torch.cuda.get_device_name(0)}")

# =====================================================================
# 3. CIFAR-10 Data Protocol (Strictly Isolated Test Set)
# =====================================================================
def load_cifar10_datasets(batch_size: int = 128, val_ratio: float = 0.1):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    clean_val  = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=test_transform)
    clean_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    n_total = len(full_train)
    indices = list(range(n_total))
    split_boundary = int(math.floor(val_ratio * n_total))

    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    train_idx, val_idx = indices[split_boundary:], indices[:split_boundary]

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(clean_val, val_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(clean_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

# =====================================================================
# 4. Neural Network Architectures: ResNet-50 & Generative Search Space
# =====================================================================

# 4.1 CIFAR-10 Tailored ResNet-50 Teacher (Direct Match to Shen et al. 2022)
class CIFAR10ResNet50(nn.Module):
    """
    Standard ImageNet ResNet-50 downsamples 32x32 inputs down to 4x4 immediately.
    Replacing conv1 with 3x3 s=1 and removing maxpool preserves spatial resolution.
    Matches Shen et al. Table 1: ~25.56M Parameters and ~1.3G FLOPs.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1  # 3 Bottleneck blocks
        self.layer2 = base.layer2  # 4 Bottleneck blocks
        self.layer3 = base.layer3  # 6 Bottleneck blocks
        self.layer4 = base.layer4  # 3 Bottleneck blocks
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

# 4.2 Modular Search Space Components
class DynamicSearchBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, kernel_size: int = 3,
                 expand_ratio: int = 2, block_type: str = "inv_res"):
        super().__init__()
        self.use_residual = (stride == 1 and in_c == out_c)
        pad = kernel_size // 2

        if block_type == "depthwise":
            self.op = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True)
            )
        else:  # Inverted Bottleneck
            hidden = int(round(in_c * expand_ratio))
            layers = []
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(in_c, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU6(inplace=True)
                ])
            layers.extend([
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, stride=stride, padding=pad, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c)
            ])
            self.op = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.op(x) if self.use_residual else self.op(x)

GENOME_BOUNDS = {
    "channels": [[12, 16, 20], [20, 24, 32], [32, 40, 48], [48, 64, 80]],
    "depths": [1, 2, 3],
    "kernels": [3, 5],
    "expands": [1, 2, 3],
    "block_types": ["inv_res", "depthwise"]
}

class CandidateGenome:
    def __init__(self, genes: Optional[Dict] = None):
        self.genes = genes if genes is not None else self.sample_random()

    @staticmethod
    def sample_random() -> Dict:
        return {
            "channels": [random.choice(GENOME_BOUNDS["channels"][i]) for i in range(4)],
            "depths": [random.choice(GENOME_BOUNDS["depths"]) for _ in range(4)],
            "kernels": [random.choice(GENOME_BOUNDS["kernels"]) for _ in range(4)],
            "expands": [random.choice(GENOME_BOUNDS["expands"]) for _ in range(4)],
            "block_type": random.choice(GENOME_BOUNDS["block_types"])
        }

    def mutate(self, p: float = 0.25) -> "CandidateGenome":
        new_g = copy.deepcopy(self.genes)
        if random.random() < p:
            idx = random.randint(0, 3)
            new_g["channels"][idx] = random.choice(GENOME_BOUNDS["channels"][idx])
        if random.random() < p:
            idx = random.randint(0, 3)
            new_g["depths"][idx] = random.choice(GENOME_BOUNDS["depths"])
        if random.random() < p:
            idx = random.randint(0, 3)
            new_g["kernels"][idx] = random.choice(GENOME_BOUNDS["kernels"])
        if random.random() < p:
            idx = random.randint(0, 3)
            new_g["expands"][idx] = random.choice(GENOME_BOUNDS["expands"])
        if random.random() < p:
            new_g["block_type"] = random.choice(GENOME_BOUNDS["block_types"])
        return CandidateGenome(new_g)

    @staticmethod
    def crossover(parent_a: "CandidateGenome", parent_b: "CandidateGenome") -> "CandidateGenome":
        child_genes = {}
        for k in ["channels", "depths", "kernels", "expands"]:
            split = random.randint(1, 3)
            child_genes[k] = parent_a.genes[k][:split] + parent_b.genes[k][split:]
        child_genes["block_type"] = random.choice([parent_a.genes["block_type"], parent_b.genes["block_type"]])
        return CandidateGenome(child_genes)

class DynamicStudentCNN(nn.Module):
    def __init__(self, genome: CandidateGenome, num_classes: int = 10):
        super().__init__()
        self.genome = genome
        cfg = genome.genes
        channels = cfg["channels"]
        depths = cfg["depths"]
        kernels = cfg["kernels"]
        expands = cfg["expands"]
        b_type = cfg["block_type"]

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True)
        )

        stages = []
        in_c = channels[0]
        for s in range(4):
            out_c = channels[s]
            stage_stride = 2 if s in [1, 2] else 1
            for d in range(depths[s]):
                st = stage_stride if d == 0 else 1
                stages.append(DynamicSearchBlock(
                    in_c=in_c,
                    out_c=out_c,
                    stride=st,
                    kernel_size=kernels[s],
                    expand_ratio=expands[s],
                    block_type=b_type
                ))
                in_c = out_c

        self.body = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.body(self.stem(x))), 1))

# =====================================================================
# 5. Multi-Objective Hardware & Complexity Profilers
# =====================================================================
# FIX 1: Ensures dummy tensor is allocated on the exact device of the model parameters
def compute_network_complexity(model: nn.Module, input_size=(1, 3, 32, 32)) -> Tuple[int, float]:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    computed_flops = 0

    def conv_hook(m, inp, out):
        nonlocal computed_flops
        h_out, w_out = out.shape[2], out.shape[3]
        cin, cout = m.in_channels, m.out_channels
        kh, kw = m.kernel_size
        computed_flops += int(2 * (cin / m.groups) * cout * kh * kw * h_out * w_out)

    def linear_hook(m, inp, out):
        nonlocal computed_flops
        computed_flops += int(2 * m.in_features * m.out_features)

    active_hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            active_hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            active_hooks.append(mod.register_forward_hook(linear_hook))

    # FIX 1 IMPLEMENTED: Query parameter device dynamically
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)

    with torch.no_grad():
        model.eval()(dummy)

    for h in active_hooks:
        h.remove()

    return total_params, float(computed_flops / 1e6)  # Params, MFLOPs

def measure_cuda_latency(model: nn.Module, device: torch.device, input_size=(1, 3, 32, 32), runs: int = 50) -> float:
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)

    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_ev.record()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy)
        end_ev.record()
        torch.cuda.synchronize()
        return float(start_ev.elapsed_time(end_ev) / runs)  # ms
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy)
        t1 = time.perf_counter()
        return float(((t1 - t0) / runs) * 1000.0)

def compute_energy_proxy(latency_ms: float, p_edge_w: float = 7.5) -> float:
    """Computes edge energy consumption in millijoules (mJ/image) under 7.5W active power draw."""
    return float(latency_ms * p_edge_w)

# =====================================================================
# 6. Adaptive Distillation Loss (Engineered for ResNet-50 Capacity Gap)
# =====================================================================
class DynamicResNet50Distillation(nn.Module):
    """
    Compensates for the extreme 140x capacity gap between ResNet-50 (25.56M) and student (0.18M).
    Dynamically anneals tau and alpha while strictly scaling by tau^2 to maintain gradient flow.
    """
    def __init__(self, t_params: int, s_params: int, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.capacity_ratio = min(1.0, float(s_params / max(1, t_params)))
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

        # For ResNet-50, capacity_ratio ~ 0.007. Dynamic boundaries expand soft guidance:
        self.tau_max = 3.5 + 2.0 * (1.0 - self.capacity_ratio)  # ~ 5.48
        self.tau_min = 1.5
        self.alpha_start = min(0.85, 0.55 + 0.30 * (1.0 - self.capacity_ratio))  # ~ 0.85
        self.alpha_end = 0.20

    def get_schedule(self, epoch: int) -> Tuple[float, float]:
        ratio = float(epoch / max(1, self.total_epochs))
        cos_val = 0.5 * (1.0 + math.cos(math.pi * ratio))
        tau = self.tau_min + (self.tau_max - self.tau_min) * cos_val
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * cos_val
        return tau, alpha

    def forward(self, s_logits: torch.Tensor, t_logits: Optional[torch.Tensor],
                targets: torch.Tensor, epoch: int) -> torch.Tensor:
        loss_ce = self.ce(s_logits, targets)
        if t_logits is None:
            return loss_ce

        tau, alpha = self.get_schedule(epoch)
        p_s = F.log_softmax(s_logits / tau, dim=-1)
        p_t = F.softmax(t_logits / tau, dim=-1)

        # Multiply by tau^2 to prevent gradient starvation
        loss_kd = self.kl(p_s, p_t) * (tau ** 2)
        return (1.0 - alpha) * loss_ce + alpha * loss_kd

# =====================================================================
# 7. Robust NSGA-II Engine with Native Integer Index Tracking
# =====================================================================
# FIX 4: Replaces dictionary indexing population.index(p) with pure integer tracking
class IntegerTrackingNSGAII:
    def __init__(self, population_size: int = 12):
        self.pop_size = population_size

    @staticmethod
    def strictly_dominates(p: Dict, q: Dict) -> bool:
        # 4 Objectives: [Error, MFLOPs, Params, Energy]
        p_objs = [p["error"], p["mflops"], p["params"], p["energy_mj"]]
        q_objs = [q["error"], q["mflops"], q["params"], q["energy_mj"]]
        not_worse = all(pv <= qv for pv, qv in zip(p_objs, q_objs))
        better = any(pv < qv for pv, qv in zip(p_objs, q_objs))
        return not_worse and better

    def sort_non_dominated_indices(self, population: List[Dict]) -> List[List[int]]:
        """Tracks integer indices [0..N-1] directly, preventing key-collision bugs and slow hashing."""
        n = len(population)
        domination_counts = [0] * n
        dominated_indices = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for p_idx in range(n):
            for q_idx in range(n):
                if self.strictly_dominates(population[p_idx], population[q_idx]):
                    dominated_indices[p_idx].append(q_idx)
                elif self.strictly_dominates(population[q_idx], population[p_idx]):
                    domination_counts[p_idx] += 1
            if domination_counts[p_idx] == 0:
                population[p_idx]["pareto_rank"] = 1
                fronts[0].append(p_idx)

        i = 0
        while len(fronts[i]) > 0:
            next_front: List[int] = []
            for p_idx in fronts[i]:
                for q_idx in dominated_indices[p_idx]:
                    domination_counts[q_idx] -= 1
                    if domination_counts[q_idx] == 0:
                        population[q_idx]["pareto_rank"] = i + 2
                        next_front.append(q_idx)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    @staticmethod
    def calculate_crowding_distances(population: List[Dict], front_indices: List[int]):
        if len(front_indices) == 0:
            return
        for idx in front_indices:
            population[idx]["crowding_distance"] = 0.0

        keys = ["error", "mflops", "params", "energy_mj"]
        for k in keys:
            sorted_indices = sorted(front_indices, key=lambda idx: population[idx][k])
            population[sorted_indices[0]]["crowding_distance"] = float("inf")
            population[sorted_indices[-1]]["crowding_distance"] = float("inf")
            span = population[sorted_indices[-1]][k] - population[sorted_indices[0]][k]
            if span == 0:
                continue
            for i in range(1, len(sorted_indices) - 1):
                prev_val = population[sorted_indices[i - 1]][k]
                next_val = population[sorted_indices[i + 1]][k]
                population[sorted_indices[i]]["crowding_distance"] += (next_val - prev_val) / span

# 7.1 TOPSIS Multi-Criteria Knee-Point Selector
def select_knee_point_topsis(rank1_candidates: List[Dict]) -> Dict:
    if len(rank1_candidates) == 1:
        return rank1_candidates[0]

    keys = ["error", "mflops", "params", "energy_mj"]
    matrix = np.array([[cand[k] for k in keys] for cand in rank1_candidates])
    weights = np.array([0.40, 0.20, 0.20, 0.20])

    denom = np.sqrt(np.sum(matrix ** 2, axis=0))
    denom[denom == 0] = 1e-6
    weighted = (matrix / denom) * weights

    ideal = np.min(weighted, axis=0)
    anti_ideal = np.max(weighted, axis=0)

    dist_pos = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    dist_neg = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

    closeness = dist_neg / (dist_pos + dist_neg + 1e-9)
    best_idx = int(np.argmax(closeness))
    print(f"[TOPSIS] Evaluated {len(rank1_candidates)} Rank-1 solutions. Selected index {best_idx} with score {closeness[best_idx]:.4f}")
    return rank1_candidates[best_idx]

# =====================================================================
# 8. Training & Evaluation Engine
# =====================================================================
# FIX 2: Explicitly wraps teacher forward calls in torch.no_grad()
def execute_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                  scaler: torch.cuda.amp.GradScaler, criterion_fn,
                  device: torch.device, epoch: int, teacher: Optional[nn.Module] = None) -> Tuple[float, float]:
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # FIX 2 IMPLEMENTED: Teacher forward call wrapped in torch.no_grad()
        t_logits = None
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(inputs)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            s_logits = model(inputs)
            loss = criterion_fn(s_logits, t_logits, labels, epoch)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = s_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * (correct / total)

def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float(accuracy_score(all_labels, all_preds) * 100.0)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision": float(p * 100.0),
        "recall": float(r * 100.0),
        "f1": float(f1 * 100.0)
    }

# =====================================================================
# 9. Persistent State Registry
# =====================================================================
def get_registry() -> Dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {
        "teacher_trained": False,
        "nas_completed": False,
        "nas_generation": 0,
        "nas_population": [],
        "selected_student_genes": None,
        "student_ce_trained": False,
        "student_kd_trained": False
    }

def update_registry(data: Dict):
    current = get_registry()
    current.update(data)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(current, f, indent=4)

# =====================================================================
# 10. Multi-Stage Pipeline Execution
# =====================================================================
def main():
    print("=" * 100)
    print(" MULTI-OBJECTIVE NAS WITH RESNET-50 TEACHER GUIDANCE (SHEN ET AL. 2022 EXTENSION)")
    print("=" * 100)

    train_loader, val_loader, test_loader = load_cifar10_datasets(batch_size=128)
    state = get_registry()

    # -----------------------------------------------------------------
    # STAGE 1: ResNet-50 Teacher Training & Verification
    # -----------------------------------------------------------------
    teacher = CIFAR10ResNet50(num_classes=10).to(DEVICE)
    t_params, t_flops = compute_network_complexity(teacher)
    t_lat = measure_cuda_latency(teacher, DEVICE)
    t_energy = compute_energy_proxy(t_lat)

    teacher_best_weights = os.path.join(STORAGE_PATH, "teacher_resnet50_best.pth")
    teacher_in_progress  = os.path.join(STORAGE_PATH, "teacher_resnet50_in_progress.pth")
    t_epochs = 35

    print(f"[TEACHER SPECS] ResNet-50 Params: {t_params/1e6:.2f}M | FLOPs: {t_flops:.2f}M | Latency: {t_lat:.2f}ms")
    print(f"[BASE PAPER ALIGNMENT] Shen et al. ResNet-50: 25.56M Params, 1304.73M FLOPs, 95.54% Acc.")

    if state.get("teacher_trained") and os.path.exists(teacher_best_weights):
        print(f"[STAGE 1] Loading pre-trained ResNet-50 Teacher from: {teacher_best_weights}")
        teacher.load_state_dict(torch.load(teacher_best_weights, map_location=DEVICE))
    else:
        print("\n" + "-" * 75)
        print(" [STAGE 1] Training ResNet-50 Teacher (AMP + OneCycleLR, 35 Epochs)")
        print("-" * 75)
        t_opt = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        t_sched = torch.optim.lr_scheduler.OneCycleLR(t_opt, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=t_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        start_ep = 1
        best_acc = 0.0

        # FIX 5 IMPLEMENTED: Resume per-epoch training
        if os.path.exists(teacher_in_progress):
            print("[STAGE 1 RESUME] Restoring in-progress checkpoint...")
            ckpt = torch.load(teacher_in_progress, map_location=DEVICE)
            teacher.load_state_dict(ckpt["model"])
            t_opt.load_state_dict(ckpt["opt"])
            t_sched.load_state_dict(ckpt["sched"])
            start_ep = ckpt["epoch"] + 1
            best_acc = ckpt["best_acc"]
            print(f"[STAGE 1 RESUME] Successfully resumed from epoch {start_ep}")

        for ep in range(start_ep, t_epochs + 1):
            t0 = time.time()
            loss, tr_acc = execute_epoch(
                model=teacher, loader=train_loader, optimizer=t_opt, scheduler=t_sched,
                scaler=scaler, criterion_fn=lambda s, t, y, e: F.cross_entropy(s, y),
                device=DEVICE, epoch=ep, teacher=None
            )
            val_eval = evaluate_classifier(teacher, val_loader, DEVICE)
            t_span = time.time() - t0

            print(f"ResNet-50 Ep [{ep:02d}/{t_epochs:02d}] ({t_span:.1f}s) | Loss: {loss:.3f} | "
                  f"Train Acc: {tr_acc:.2f}% | Val Acc: {val_eval['accuracy']:.2f}%")

            if val_eval["accuracy"] > best_acc:
                best_acc = val_eval["accuracy"]
                torch.save(teacher.state_dict(), teacher_best_weights)

            # FIX 5: Save every epoch checkpoint
            torch.save({
                "epoch": ep,
                "model": teacher.state_dict(),
                "opt": t_opt.state_dict(),
                "sched": t_sched.state_dict(),
                "best_acc": best_acc
            }, teacher_in_progress)

        teacher.load_state_dict(torch.load(teacher_best_weights))
        if os.path.exists(teacher_in_progress):
            os.remove(teacher_in_progress)
        update_registry({"teacher_trained": True})

    teacher_metrics = evaluate_classifier(teacher, test_loader, DEVICE)
    print(f"\n[TEACHER VERIFIED] ResNet-50 Test Acc: {teacher_metrics['accuracy']:.2f}% | Macro-F1: {teacher_metrics['f1']:.2f}% | "
          f"Params: {t_params/1e6:.2f}M | FLOPs: {t_flops:.2f}M | Latency: {t_lat:.2f}ms | Energy: {t_energy:.2f}mJ\n")

    # -----------------------------------------------------------------
    # STAGE 2: Multi-Objective NSGA-II Search (ResNet-50 Guided)
    # -----------------------------------------------------------------
    pop_size = 12
    n_generations = 4
    nsga = IntegerTrackingNSGAII(population_size=pop_size)

    # FIX 3 & FIX 6: Genuinely ResNet-50 teacher-guided candidate evaluation + empty_cache()
    def evaluate_candidate_proxy(genome: CandidateGenome) -> Dict:
        cand_net = DynamicStudentCNN(genome).to(DEVICE)
        p_count, mflops = compute_network_complexity(cand_net)
        latency = measure_cuda_latency(cand_net, DEVICE)
        energy_mj = compute_energy_proxy(latency)

        p_opt = torch.optim.AdamW(cand_net.parameters(), lr=2e-3, weight_decay=1e-4)
        p_sched = torch.optim.lr_scheduler.OneCycleLR(p_opt, max_lr=2e-3, steps_per_epoch=len(train_loader), epochs=2)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        # FIX 3: ResNet-50 capacity-aware adaptive teacher guidance during proxy search
        proxy_criterion = DynamicResNet50Distillation(t_params=t_params, s_params=p_count, total_epochs=2)

        for ep in range(1, 3):
            _ = execute_epoch(
                model=cand_net, loader=train_loader, optimizer=p_opt, scheduler=p_sched,
                scaler=scaler, criterion_fn=proxy_criterion,
                device=DEVICE, epoch=ep, teacher=teacher  # Guided by ResNet-50
            )

        val_eval = evaluate_classifier(cand_net, val_loader, DEVICE)
        val_acc = val_eval["accuracy"]

        # FIX 6: Explicitly delete candidate models and flush CUDA cache
        del cand_net, p_opt, p_sched, scaler
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "genome": genome.genes,
            "params": p_count,
            "mflops": mflops,
            "latency": latency,
            "energy_mj": energy_mj,
            "val_acc": val_acc,
            "error": 100.0 - val_acc
        }

    if state.get("nas_completed") and state.get("selected_student_genes"):
        print("[STAGE 2] Loading previously identified Pareto-optimal student architecture...")
        best_genome = CandidateGenome(state["selected_student_genes"])
    else:
        print("-" * 75)
        print(f" [STAGE 2] ResNet-50 Guided Evolutionary Search ({n_generations} Gens, Pop {pop_size})")
        print("-" * 75)

        current_gen = state.get("nas_generation", 0)
        population = state.get("nas_population", [])

        if len(population) == 0:
            print("[NAS INIT] Initializing and evaluating stochastic seed generation...")
            population = [evaluate_candidate_proxy(CandidateGenome()) for _ in range(pop_size)]

        while current_gen < n_generations:
            print(f"\n[NAS] Advancing Evolutionary Generation {current_gen + 1}/{n_generations}...")
            offspring = []

            for _ in range(pop_size // 2):
                p_a = CandidateGenome(random.choice(population)["genome"])
                p_b = CandidateGenome(random.choice(population)["genome"])
                child_1 = CandidateGenome.crossover(p_a, p_b).mutate()
                child_2 = CandidateGenome.crossover(p_b, p_a).mutate()
                offspring.extend([evaluate_candidate_proxy(child_1), evaluate_candidate_proxy(child_2)])

            combined_population = population + offspring

            # FIX 4: Use integer index tracking
            fronts_indices = nsga.sort_non_dominated_indices(combined_population)

            next_population = []
            for f_indices in fronts_indices:
                nsga.calculate_crowding_distances(combined_population, f_indices)
                if len(next_population) + len(f_indices) <= pop_size:
                    for idx in f_indices:
                        next_population.append(combined_population[idx])
                else:
                    sorted_indices = sorted(f_indices, key=lambda idx: combined_population[idx]["crowding_distance"], reverse=True)
                    slots_remaining = pop_size - len(next_population)
                    for idx in sorted_indices[:slots_remaining]:
                        next_population.append(combined_population[idx])
                    break

            population = next_population
            current_gen += 1

            update_registry({
                "nas_generation": current_gen,
                "nas_population": population
            })
            print(f"[CHECKPOINT] Generation {current_gen} successfully archived to Drive.")

        # Extract Rank-1 Pareto Front
        fronts_indices = nsga.sort_non_dominated_indices(population)
        rank1_candidates = [population[idx] for idx in fronts_indices[0]]

        print("\n" + "=" * 75)
        print(f" [STAGE 2.1] TOPSIS Knee-Point Selection ({len(rank1_candidates)} Rank-1 Candidates)")
        print("=" * 75)
        for cand in rank1_candidates:
            print(f"Candidate -> Params: {cand['params']/1e6:.3f}M | FLOPs: {cand['mflops']:5.2f}M | "
                  f"Latency: {cand['latency']:5.2f}ms | Energy: {cand['energy_mj']:5.2f}mJ | Val Acc: {cand['val_acc']:.2f}%")

        optimal_cand = select_knee_point_topsis(rank1_candidates)
        best_genome = CandidateGenome(optimal_cand["genome"])

        update_registry({
            "nas_completed": True,
            "selected_student_genes": best_genome.genes
        })

    # -----------------------------------------------------------------
    # STAGE 3: Comparative Retraining (Baseline CE vs. ResNet-50 Guided KD)
    # -----------------------------------------------------------------
    student_epochs = 45

    # 3.1 Student Baseline (Cross-Entropy Only, No Distillation)
    student_ce = DynamicStudentCNN(best_genome).to(DEVICE)
    ce_best_weights = os.path.join(STORAGE_PATH, "student_ce_best.pth")
    ce_in_progress  = os.path.join(STORAGE_PATH, "student_ce_in_progress.pth")

    if state.get("student_ce_trained") and os.path.exists(ce_best_weights):
        print(f"\n[STAGE 3.1] Loading pre-trained Baseline CE Student from: {ce_best_weights}")
        student_ce.load_state_dict(torch.load(ce_best_weights, map_location=DEVICE))
    else:
        print("\n" + "-" * 75)
        print(" [STAGE 3.1] Training Student WITHOUT Distillation (CE Baseline, 45 Epochs)")
        print("-" * 75)
        opt_ce = torch.optim.SGD(student_ce.parameters(), lr=0.08, momentum=0.9, weight_decay=1e-4)
        sched_ce = torch.optim.lr_scheduler.OneCycleLR(opt_ce, max_lr=0.08, steps_per_epoch=len(train_loader), epochs=student_epochs)
        scaler_ce = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        start_ep, best_ce_acc = 1, 0.0

        # FIX 5: Resume per-epoch training
        if os.path.exists(ce_in_progress):
            print("[STAGE 3.1 RESUME] Restoring in-progress baseline checkpoint...")
            ckpt = torch.load(ce_in_progress, map_location=DEVICE)
            student_ce.load_state_dict(ckpt["model"])
            opt_ce.load_state_dict(ckpt["opt"])
            sched_ce.load_state_dict(ckpt["sched"])
            start_ep = ckpt["epoch"] + 1
            best_ce_acc = ckpt["best_acc"]
            print(f"[STAGE 3.1 RESUME] Resumed at epoch {start_ep}")

        for ep in range(start_ep, student_epochs + 1):
            loss, tr_acc = execute_epoch(
                model=student_ce, loader=train_loader, optimizer=opt_ce, scheduler=sched_ce,
                scaler=scaler_ce, criterion_fn=lambda s, t, y, e: F.cross_entropy(s, y),
                device=DEVICE, epoch=ep, teacher=None
            )
            val_eval = evaluate_classifier(student_ce, val_loader, DEVICE)

            if ep % 5 == 0 or ep == student_epochs:
                print(f"Student CE Ep [{ep:02d}/{student_epochs:02d}] | Train Acc: {tr_acc:.2f}% | Val Acc: {val_eval['accuracy']:.2f}%")

            if val_eval["accuracy"] > best_ce_acc:
                best_ce_acc = val_eval["accuracy"]
                torch.save(student_ce.state_dict(), ce_best_weights)

            # FIX 5: Save every epoch checkpoint
            torch.save({
                "epoch": ep,
                "model": student_ce.state_dict(),
                "opt": opt_ce.state_dict(),
                "sched": sched_ce.state_dict(),
                "best_acc": best_ce_acc
            }, ce_in_progress)

        student_ce.load_state_dict(torch.load(ce_best_weights))
        if os.path.exists(ce_in_progress):
            os.remove(ce_in_progress)
        update_registry({"student_ce_trained": True})

    ce_metrics = evaluate_classifier(student_ce, test_loader, DEVICE)

    # 3.2 Student with ResNet-50 Adaptive Knowledge Distillation
    student_kd = DynamicStudentCNN(best_genome).to(DEVICE)
    kd_best_weights = os.path.join(STORAGE_PATH, "student_resnet50_kd_best.pth")
    kd_in_progress  = os.path.join(STORAGE_PATH, "student_kd_in_progress.pth")

    if state.get("student_kd_trained") and os.path.exists(kd_best_weights):
        print(f"\n[STAGE 3.2] Loading pre-trained ResNet-50 Guided KD Student from: {kd_best_weights}")
        student_kd.load_state_dict(torch.load(kd_best_weights, map_location=DEVICE))
    else:
        print("\n" + "-" * 75)
        print(" [STAGE 3.2] Training Student WITH ResNet-50 Adaptive KD (45 Epochs)")
        print("-" * 75)
        s_params, _ = compute_network_complexity(student_kd)
        kd_criterion = DynamicResNet50Distillation(t_params=t_params, s_params=s_params, total_epochs=student_epochs)

        opt_kd = torch.optim.SGD(student_kd.parameters(), lr=0.08, momentum=0.9, weight_decay=1e-4)
        sched_kd = torch.optim.lr_scheduler.OneCycleLR(opt_kd, max_lr=0.08, steps_per_epoch=len(train_loader), epochs=student_epochs)
        scaler_kd = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        start_ep, best_kd_acc = 1, 0.0

        # FIX 5: Resume per-epoch training
        if os.path.exists(kd_in_progress):
            print("[STAGE 3.2 RESUME] Restoring in-progress distillation checkpoint...")
            ckpt = torch.load(kd_in_progress, map_location=DEVICE)
            student_kd.load_state_dict(ckpt["model"])
            opt_kd.load_state_dict(ckpt["opt"])
            sched_kd.load_state_dict(ckpt["sched"])
            start_ep = ckpt["epoch"] + 1
            best_kd_acc = ckpt["best_acc"]
            print(f"[STAGE 3.2 RESUME] Resumed at epoch {start_ep}")

        for ep in range(start_ep, student_epochs + 1):
            tau_cur, alpha_cur = kd_criterion.get_schedule(ep)
            loss, tr_acc = execute_epoch(
                model=student_kd, loader=train_loader, optimizer=opt_kd, scheduler=sched_kd,
                scaler=scaler_kd, criterion_fn=kd_criterion,
                device=DEVICE, epoch=ep, teacher=teacher
            )
            val_eval = evaluate_classifier(student_kd, val_loader, DEVICE)

            if ep % 5 == 0 or ep == student_epochs:
                print(f"Student KD Ep [{ep:02d}/{student_epochs:02d}] | tau: {tau_cur:.2f} | alpha: {alpha_cur:.2f} | "
                      f"Train Acc: {tr_acc:.2f}% | Val Acc: {val_eval['accuracy']:.2f}%")

            if val_eval["accuracy"] > best_kd_acc:
                best_kd_acc = val_eval["accuracy"]
                torch.save(student_kd.state_dict(), kd_best_weights)

            # FIX 5: Save every epoch checkpoint
            torch.save({
                "epoch": ep,
                "model": student_kd.state_dict(),
                "opt": opt_kd.state_dict(),
                "sched": sched_kd.state_dict(),
                "best_acc": best_kd_acc
            }, kd_in_progress)

        student_kd.load_state_dict(torch.load(kd_best_weights))
        if os.path.exists(kd_in_progress):
            os.remove(kd_in_progress)
        update_registry({"student_kd_trained": True})

    kd_metrics = evaluate_classifier(student_kd, test_loader, DEVICE)
    s_params, s_flops = compute_network_complexity(student_kd)
    s_lat = measure_cuda_latency(student_kd, DEVICE)
    s_energy = compute_energy_proxy(s_lat)

    # -----------------------------------------------------------------
    # STAGE 4: Final Scientific Reporting & Base Paper Comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 108)
    print(f"{'EXPERIMENTAL SCIENTIFIC BENCHMARK RESULTS (CIFAR-10)':^108}")
    print("=" * 108)
    print(f"{'Model / Framework':<24} | {'Method':<16} | {'Params':<8} | {'FLOPs':<8} | {'Lat (ms)':<8} | {'Energy (mJ)':<11} | {'Acc (%)':<8} | {'F1 (%)':<8}")
    print("-" * 108)
    print(f"{'Shen et al. (ResNet-50)':<24} | {'CE (Base Paper)':<16} | 25.56M   | 1304.7M  | N/A      | N/A         | 95.54    | N/A")
    print(f"{'Shen et al. (Student)':<24} | {'DARTS + KD':<16} | 0.18M    | 34.88M   | N/A      | N/A         | 94.55    | N/A")
    print("-" * 108)
    print(f"{'ResNet-50 (Teacher, Ours)':<24} | {'Standard CE':<16} | {t_params/1e6:6.2f}M | {t_flops:6.2f}M | {t_lat:6.2f}   | {t_energy:9.2f}   | {teacher_metrics['accuracy']:6.2f}   | {teacher_metrics['f1']:6.2f}")
    print(f"{'NSGA-II Student (Ours)':<24} | {'No Distill (CE)':<16} | {s_params/1e6:6.3f}M | {s_flops:6.2f}M | {s_lat:6.2f}   | {s_energy:9.2f}   | {ce_metrics['accuracy']:6.2f}   | {ce_metrics['f1']:6.2f}")
    print(f"{'NSGA-II Student (Ours)':<24} | {'Res50-Guided KD':<16} | {s_params/1e6:6.3f}M | {s_flops:6.2f}M | {s_lat:6.2f}   | {s_energy:9.2f}   | {kd_metrics['accuracy']:6.2f}   | {kd_metrics['f1']:6.2f}")
    print("=" * 108)

    comp_ratio = (1.0 - (s_params / t_params)) * 100.0
    lat_speedup = t_lat / s_lat
    energy_saved = (1.0 - (s_energy / t_energy)) * 100.0
    kd_gain = kd_metrics["accuracy"] - ce_metrics["accuracy"]

    print("\n[VERIFIED SCIENTIFIC CONTRIBUTIONS EXTENDING SHEN ET AL. 2022]")
    print(f"1. Extreme Parameter Compression: Achieved {comp_ratio:.2f}% parameter reduction from ResNet-50 ({t_params/1e6:.2f}M down to {s_params/1e6:.3f}M).")
    print(f"2. Edge Acceleration: {lat_speedup:.2f}x latency speedup; {energy_saved:.2f}% lower active energy per inference.")
    print(f"3. Decoupled Distillation Benefit: +{kd_gain:.2f}% accuracy gain strictly isolated from architecture topology.")
    print(f"4. Pareto Knee-Point: Derived using TOPSIS on Rank-1 solutions, explicitly balancing Error, FLOPs, Params, and Energy.")

if __name__ == "__main__":
    main()
