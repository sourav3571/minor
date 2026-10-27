import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


SPACE = {
    "depth": [1, 2, 3],
    "width": [8, 16, 24, 32],
    "operation": ["regular", "depthwise"],
    "kernel": [3, 5],
}


def validate_config(config):
    for key, allowed in SPACE.items():
        if config[key] not in allowed:
            raise ValueError(f"invalid {key}={config[key]}; allowed={allowed}")


def complexity(config):
    return config["depth"] * config["width"] ** 2


def adaptive_alpha(config):
    values = [depth * width ** 2 for depth in SPACE["depth"] for width in SPACE["width"]]
    low, high = min(values), max(values)
    normalized = (complexity(config) - low) / (high - low)
    alpha = 0.2 + 0.6 * normalized
    return alpha, normalized


def pareto_dominates(left, right):
    no_worse = (left["accuracy"] >= right["accuracy"] and
                left["latency_ms"] <= right["latency_ms"] and
                left["state_dict_bytes"] <= right["state_dict_bytes"])
    strictly_better = (left["accuracy"] > right["accuracy"] or
                       left["latency_ms"] < right["latency_ms"] or
                       left["state_dict_bytes"] < right["state_dict_bytes"])
    return no_worse and strictly_better


def pareto_front(rows):
    return [row for row in rows if not any(
        other is not row and pareto_dominates(other, row) for other in rows
    )]


def mutate(config):
    child = dict(config)
    key = random.choice(list(SPACE))
    alternatives = [value for value in SPACE[key] if value != child[key]]
    child[key] = random.choice(alternatives)
    validate_config(child)
    return child


class StudentCNN(nn.Module):
    def __init__(self, config, classes=10):
        super().__init__()
        validate_config(config)
        blocks = []
        incoming = 3
        for _ in range(config["depth"]):
            width, kernel = config["width"], config["kernel"]
            padding = kernel // 2
            if config["operation"] == "depthwise":
                blocks.extend([
                    nn.Conv2d(incoming, incoming, kernel, padding=padding,
                              groups=incoming, bias=False),
                    nn.Conv2d(incoming, width, 1, bias=False),
                    nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
                ])
            else:
                blocks.extend([
                    nn.Conv2d(incoming, width, kernel, padding=padding, bias=False),
                    nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
                ])
            incoming = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(incoming, classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(self.pool(features).flatten(1))


class TeacherCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        blocks = []
        incoming = 3
        for width in (32, 64, 128):
            blocks.extend([nn.Conv2d(incoming, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
            incoming = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(incoming, classes)

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)).flatten(1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(args):
    dataset_type = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    classes = 10 if args.dataset == "cifar10" else 100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        if args.dataset == "cifar10" else transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),
    ])
    full = dataset_type(args.data_dir, train=True, download=True, transform=transform)
    test = dataset_type(args.data_dir, train=False, download=True, transform=transform)
    if args.train_limit:
        full = Subset(full, range(min(args.train_limit, len(full))))
    if args.test_limit:
        test = Subset(test, range(min(args.test_limit, len(test))))
    validation_size = max(1, int(0.2 * len(full)))
    train_set, validation_set = random_split(
        full, [len(full) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    generator = torch.Generator().manual_seed(args.seed)
    common = {"batch_size": args.batch_size, "num_workers": 0,
              "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation_set, shuffle=False, **common)
    test_loader = DataLoader(test, shuffle=False, **common)
    return train_loader, validation_loader, test_loader, classes, len(train_set), len(validation_set)


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.numel()
    return {"loss": total_loss / count, "accuracy": correct / count}


def train_model(model, train_loader, validation_loader, device, epochs, learning_rate,
                teacher=None, alpha=1.0, temperature=4.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_state, best_accuracy = [], None, -1.0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count, latest_ce, latest_kl = 0.0, 0, 0, 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(images)
            ce = F.cross_entropy(student_logits, labels)
            if teacher is None or alpha == 1.0:
                loss = ce
                kl = torch.zeros((), device=device)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                if teacher_logits.shape != student_logits.shape:
                    raise ValueError("teacher and student class logits must have identical shapes")
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                loss = alpha * ce + (1 - alpha) * temperature ** 2 * kl
            loss.backward()
            optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (student_logits.argmax(1) == labels).sum().item()
            latest_ce, latest_kl = ce.item(), kl.item()
        validation = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": total_loss / count,
                        "train_accuracy": correct / count, "validation": validation,
                        "last_batch_ce": latest_ce, "last_batch_kl_unscaled": latest_kl})
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        print(f"epoch={epoch} train_loss={total_loss/count:.4f} "
              f"val_acc={validation['accuracy']:.3f}")
    model.load_state_dict(best_state)
    return model, history, time.perf_counter() - start


def measure_latency(model, loader, device, repeats=80):
    model.eval()
    sample = next(iter(loader))[0][:1].to(device)
    with torch.no_grad():
        for _ in range(15):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "repeats": repeats}


def main():
    parser = argparse.ArgumentParser(description="Adaptive teacher-guided multi-objective CIFAR NAS")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--guidance", choices=["adaptive", "fixed", "none"], default="adaptive")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--fixed-alpha", type=float, default=0.5)
    parser.add_argument("--teacher-epochs", type=int, default=2)
    parser.add_argument("--student-epochs", type=int, default=2)
    parser.add_argument("--initial-population", type=int, default=3)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/project5"))
    args = parser.parse_args()
    if args.temperature <= 0 or not 0 <= args.fixed_alpha <= 1:
        parser.error("temperature must be > 0 and fixed-alpha in [0,1]")
    if min(args.teacher_epochs, args.student_epochs, args.initial_population) < 1 or args.generations < 0:
        parser.error("epochs and initial-population must be positive; generations cannot be negative")

    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader, classes, train_n, validation_n = make_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TeacherCNN(classes)
    print(f"Training {args.dataset} teacher on {device}")
    teacher, teacher_history, teacher_seconds = train_model(
        teacher, train_loader, validation_loader, device, args.teacher_epochs, args.learning_rate
    )
    teacher_path = args.output_dir / "teacher.pt"
    torch.save({"state_dict": teacher.state_dict(), "classes": classes,
                "dataset": args.dataset}, teacher_path)
    teacher.eval()
    teacher.requires_grad_(False)

    population = []
    while len(population) < args.initial_population:
        config = {key: random.choice(values) for key, values in SPACE.items()}
        if config not in population:
            population.append(config)

    records = []
    archive = []
    candidate_id = 0
    generations = args.generations + 1
    for generation in range(generations):
        for config in population:
            validate_config(config)
            alpha, normalized_complexity = adaptive_alpha(config)
            if args.guidance == "fixed":
                alpha = args.fixed_alpha
            elif args.guidance == "none":
                alpha = 1.0
            seed_all(args.seed + candidate_id + 1)
            train_loader.generator.manual_seed(args.seed)
            model = StudentCNN(config, classes)
            model, history, training_seconds = train_model(
                model, train_loader, validation_loader, device, args.student_epochs,
                args.learning_rate, teacher=teacher, alpha=alpha,
                temperature=args.temperature
            )
            validation = evaluate(model, validation_loader, device)
            latency = measure_latency(model, validation_loader, device)
            path = args.output_dir / f"candidate_{candidate_id:04d}.pt"
            torch.save({"config": config, "state_dict": model.state_dict(),
                        "classes": classes, "dataset": args.dataset}, path)
            row = {"candidate_id": candidate_id, "generation": generation,
                   "config": config, "complexity_proxy": complexity(config),
                   "normalized_complexity": normalized_complexity, "alpha_ce": alpha,
                   "kd_weight": 1 - alpha, "guidance_mode": args.guidance,
                   "validation_accuracy": validation["accuracy"],
                   "validation_loss": validation["loss"],
                   "latency_ms": latency["median_ms"],
                   "latency_p95_ms": latency["p95_ms"],
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "state_dict_bytes": path.stat().st_size,
                   "energy_joules": None, "energy_status": "not_measured",
                   "training_seconds": training_seconds, "history": history}
            records.append(row)
            objective_record = {"candidate_id": candidate_id, "config": config,
                                "accuracy": row["validation_accuracy"],
                                "latency_ms": row["latency_ms"],
                                "state_dict_bytes": row["state_dict_bytes"],
                                "path": str(path)}
            archive = pareto_front(archive + [objective_record])
            print(f"candidate={candidate_id} gen={generation} config={config} alpha={alpha:.3f} "
                  f"val_acc={validation['accuracy']:.3f} latency_ms={latency['median_ms']:.3f} "
                  f"bytes={path.stat().st_size}")
            candidate_id += 1
        if generation < args.generations:
            population = [mutate(random.choice(archive)["config"])
                          for _ in range(args.initial_population)]

    selected = max(archive, key=lambda item: item["accuracy"])
    selected_config = selected["config"]
    checkpoint = torch.load(selected["path"], map_location=device, weights_only=True)
    final_model = StudentCNN(selected_config, classes).to(device)
    final_model.load_state_dict(checkpoint["state_dict"])
    final_test = evaluate(final_model, test_loader, device)
    print(f"FINAL TEST candidate={selected['candidate_id']} accuracy={final_test['accuracy']:.3f} "
          f"loss={final_test['loss']:.4f}")

    figure, axis = plt.subplots(figsize=(7, 5))
    for row in records:
        axis.scatter(row["latency_ms"], row["validation_accuracy"],
                     marker="o" if any(item["candidate_id"] == row["candidate_id"] for item in archive) else "x")
        axis.annotate(str(row["candidate_id"]),
                      (row["latency_ms"], row["validation_accuracy"]))
    axis.set(title="Measured local validation accuracy vs batch-1 latency",
             xlabel="Local latency (ms; lower is better)",
             ylabel="Validation accuracy (higher is better)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "candidate_tradeoff.png", dpi=150)
    plt.close(figure)

    result = {"arguments": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
              "protocol": {"device": str(device), "torch_version": torch.__version__,
                           "train_examples": train_n, "validation_examples": validation_n,
                           "test_examples": len(test_loader.dataset),
                           "latency": "batch 1, 15 warmups, 80 repetitions, median and p95; forward only",
                           "energy": "not measured; use target-board power instrumentation",
                           "selection": "Pareto archive then highest validation accuracy; test used once"},
              "teacher": {"validation": evaluate(teacher, validation_loader, device),
                          "training_seconds": teacher_seconds,
                          "history": teacher_history,
                          "checkpoint_bytes": teacher_path.stat().st_size},
              "candidates": records,
              "pareto_archive": [{key: value for key, value in item.items() if key != "path"}
                                 for item in archive],
              "selected_candidate_id": selected["candidate_id"],
              "selected_config": selected_config,
              "test_result": final_test}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
