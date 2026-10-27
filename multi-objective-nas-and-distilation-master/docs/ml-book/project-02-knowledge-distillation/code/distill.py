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


class SmallCNN(nn.Module):
    def __init__(self, channels=16, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(channels * 2 * 7 * 7, classes)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


class TeacherCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128),
                                        nn.ReLU(), nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loaders(args):
    dataset_type = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    full = dataset_type(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    test = dataset_type(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.limit:
        full = Subset(full, range(min(args.limit, len(full))))
        test = Subset(test, range(min(args.test_limit, len(test))))
    val_n = max(1, int(0.2 * len(full)))
    train_set, val_set = random_split(full, [len(full) - val_n, val_n],
                                      generator=torch.Generator().manual_seed(args.seed))
    generator = torch.Generator().manual_seed(args.seed)
    opts = {"batch_size": args.batch_size, "num_workers": 0,
            "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train_set, shuffle=True, generator=generator, **opts),
            DataLoader(val_set, shuffle=False, **opts),
            DataLoader(test, shuffle=False, **opts))


def evaluate(model, loader, device):
    model.eval()
    correct, count, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.numel()
    return {"loss": total_loss / count, "accuracy": correct / count}


def train(model, train_loader, val_loader, device, epochs, learning_rate,
          teacher=None, temperature=4.0, alpha=0.5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(images)
            ce = F.cross_entropy(student_logits, labels)
            if teacher is None:
                loss = ce
                kd_value = 0.0
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                if teacher_logits.shape != student_logits.shape:
                    raise ValueError(f"teacher/student output mismatch: {teacher_logits.shape} vs "
                                     f"{student_logits.shape}")
                student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
                teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
                kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
                loss = alpha * ce + (1.0 - alpha) * (temperature ** 2) * kd
                kd_value = kd.item()
            loss.backward()
            optimizer.step()
            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            correct += (student_logits.argmax(1) == labels).sum().item()
            count += batch_n
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / count,
               "train_accuracy": correct / count, "validation": validation,
               "last_batch_kd_unscaled": kd_value}
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} "
              f"train_acc={row['train_accuracy']:.3f} val_acc={validation['accuracy']:.3f}")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def measure(model, loader, device, repeats=100):
    model.eval()
    images, _ = next(iter(loader))
    sample = images[:1].to(device)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def parameter_count(model):
    return sum(value.numel() for value in model.parameters())


def save_model(model, path, metadata):
    torch.save({"state_dict": model.state_dict(), **metadata}, path)
    return path.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/project2"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs/batch-size must be positive; learning-rate must be > 0")
    if args.temperature <= 0 or not 0 <= args.alpha <= 1:
        parser.error("temperature must be > 0 and alpha must be in [0, 1]")

    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TeacherCNN().to(device)
    print("Training teacher")
    teacher, teacher_history = train(teacher, train_loader, val_loader, device, args.epochs,
                                     args.learning_rate)
    teacher_path = args.output_dir / "teacher.pt"
    teacher_bytes = save_model(teacher, teacher_path, {"classes": 10})
    teacher.eval()
    example_images, example_labels = next(iter(val_loader))
    with torch.no_grad():
        example_logits = teacher(example_images[:1].to(device))
        example_probabilities = F.softmax(example_logits, dim=1)
    print(f"teacher inspection: true_label={example_labels[0].item()} "
          f"logits={example_logits[0].cpu().tolist()} "
          f"probabilities={example_probabilities[0].cpu().tolist()}")
    teacher.requires_grad_(False)

    results = {}
    models = {"teacher": teacher}
    for name, use_teacher in [("student_ce", False), ("student_kd", True)]:
        seed_all(args.seed)
        train_loader.generator.manual_seed(args.seed)
        student = SmallCNN().to(device)
        print(f"Training {name}")
        student, history = train(student, train_loader, val_loader, device, args.epochs,
                                 args.learning_rate,
                                 teacher=teacher if use_teacher else None,
                                 temperature=args.temperature, alpha=args.alpha)
        path = args.output_dir / f"{name}.pt"
        bytes_on_disk = save_model(student, path, {"classes": 10, "seed": args.seed})
        validation = evaluate(student, val_loader, device)
        latency_ms = measure(student, val_loader, device)
        with torch.no_grad():
            example_logits = student(example_images[:1].to(device))
            example_probabilities = F.softmax(example_logits, dim=1)
        results[name] = {"history": history, "validation": validation,
                         "parameters": parameter_count(student),
                         "checkpoint_bytes": bytes_on_disk, "latency_ms_per_image": latency_ms,
                         "example_true_label": int(example_labels[0]),
                         "example_logits": example_logits[0].cpu().tolist(),
                         "example_probabilities": example_probabilities[0].cpu().tolist(),
                         "energy": "not_measured"}
        print(f"{name} inspection: true_label={example_labels[0].item()} "
              f"logits={example_logits[0].cpu().tolist()} "
              f"probabilities={example_probabilities[0].cpu().tolist()}")
        models[name] = student

    test_results = {name: evaluate(model, test_loader, device) for name, model in models.items()}
    for name, metrics in test_results.items():
        print(f"FINAL TEST {name}: accuracy={metrics['accuracy']:.3f} loss={metrics['loss']:.4f}")
    results["teacher"] = {"history": teacher_history, "validation": evaluate(teacher, val_loader, device),
                          "parameters": parameter_count(teacher), "checkpoint_bytes": teacher_bytes,
                          "latency_ms_per_image": measure(teacher, val_loader, device),
                          "energy": "not_measured"}
    results["test"] = test_results
    results["protocol"] = {"device": str(device), "torch_version": torch.__version__,
                           "epochs": args.epochs, "seed": args.seed,
                           "temperature": args.temperature, "alpha": args.alpha,
                           "latency": "batch-1, 20 warmups, 100 repeated forwards; local device",
                           "energy": "not measured; requires external instrument on target board"}
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved results to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
