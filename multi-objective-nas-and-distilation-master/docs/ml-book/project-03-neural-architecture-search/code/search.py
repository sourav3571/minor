import argparse
import json
import random
import time

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CHOICES = {
    "depth": [1, 2, 3],
    "channels": [8, 16, 24, 32],
    "kernel": [3, 5],
}


def validate_config(config):
    if config["depth"] not in CHOICES["depth"]:
        raise ValueError("depth must be one of the allowed choices")
    if config["channels"] not in CHOICES["channels"]:
        raise ValueError("channels must be one of the allowed choices")
    if config["kernel"] not in CHOICES["kernel"]:
        raise ValueError("kernel must be one of the allowed choices")


class CandidateCNN(nn.Module):
    def __init__(self, config, classes=10):
        super().__init__()
        validate_config(config)
        layers = []
        in_channels = 1
        for _ in range(config["depth"]):
            layers.extend([
                nn.Conv2d(in_channels, config["channels"], config["kernel"], padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_channels = config["channels"]
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            shape = self.features(torch.zeros(1, 1, 28, 28)).shape
        self.classifier = nn.Linear(int(np.prod(shape[1:])), classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features.flatten(1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(args):
    dataset_type = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    full = dataset_type(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    test = dataset_type(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.train_limit:
        full = Subset(full, range(min(args.train_limit, len(full))))
    if args.test_limit:
        test = Subset(test, range(min(args.test_limit, len(test))))
    val_n = max(1, int(0.2 * len(full)))
    train_set, val_set = random_split(full, [len(full) - val_n, val_n],
                                      generator=torch.Generator().manual_seed(args.seed))
    loader_generator = torch.Generator().manual_seed(args.seed)
    common = {"batch_size": args.batch_size, "num_workers": 0}
    return (DataLoader(train_set, shuffle=True, generator=loader_generator, **common),
            DataLoader(val_set, shuffle=False, **common), DataLoader(test, shuffle=False, **common))


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


def train_candidate(config, train_loader, val_loader, device, args):
    model = CandidateCNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_accuracy = -1.0
    best_state = None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / count, "validation": val})
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    return model, history, training_seconds


def latency_ms(model, loader, device, repeats=50):
    model.eval()
    sample = next(iter(loader))[0][:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/project3"))
    args = parser.parse_args()
    if args.candidates < 1 or args.epochs < 1:
        parser.error("candidates and epochs must be positive")
    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = make_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    winner = None
    winner_accuracy = -1.0

    for candidate_id in range(args.candidates):
        config = {name: random.choice(values) for name, values in CHOICES.items()}
        validate_config(config)
        seed_all(args.seed + candidate_id)
        train_loader.generator.manual_seed(args.seed)
        model, history, train_seconds = train_candidate(
            config, train_loader, validation_loader, device, args
        )
        validation = evaluate(model, validation_loader, device)
        count = sum(value.numel() for value in model.parameters())
        path = args.output_dir / f"candidate_{candidate_id:03d}.pt"
        torch.save({"config": config, "state_dict": model.state_dict()}, path)
        record = {"candidate_id": candidate_id, "config": config,
                  "validation": validation, "parameters": count,
                  "checkpoint_bytes": path.stat().st_size,
                  "latency_ms_per_image": latency_ms(model, validation_loader, device),
                  "training_seconds": train_seconds, "history": history}
        records.append(record)
        print(f"candidate={candidate_id} config={config} val_acc={validation['accuracy']:.3f} "
              f"params={count} latency_ms={record['latency_ms_per_image']:.3f}")
        if validation["accuracy"] > winner_accuracy:
            winner_accuracy = validation["accuracy"]
            winner = {"record": record, "model": model, "path": path}

    final_test = evaluate(winner["model"], test_loader, device)
    figure, axis = plt.subplots(figsize=(7, 5))
    for record in records:
        axis.scatter(record["latency_ms_per_image"], record["validation"]["accuracy"])
        axis.annotate(str(record["candidate_id"]),
                      (record["latency_ms_per_image"], record["validation"]["accuracy"]))
    axis.set(title="Random NAS candidates: validation accuracy vs local latency",
             xlabel="Local batch-1 latency (ms; lower is better)",
             ylabel="Validation accuracy (higher is better)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "search_tradeoff.png", dpi=150)
    plt.close(figure)
    print(f"FINAL TEST winner_id={winner['record']['candidate_id']} "
          f"accuracy={final_test['accuracy']:.3f} loss={final_test['loss']:.4f}")
    result = {"protocol": {"device": str(device), "seed": args.seed,
                            "epochs_per_candidate": args.epochs,
                            "candidate_budget": args.candidates,
                            "torch_version": torch.__version__,
                            "test_used_only_after_selection": True},
              "candidates": records,
              "winner": {"candidate_id": winner["record"]["candidate_id"],
                         "config": winner["record"]["config"],
                         "validation": winner["record"]["validation"],
                         "test": final_test}}
    (args.output_dir / "search_results.json").write_text(json.dumps(result, indent=2))
    (args.output_dir / "winner_config.json").write_text(json.dumps(winner["record"]["config"], indent=2))
    print(f"winner_config={args.output_dir / 'winner_config.json'}")
    print(f"full_log={args.output_dir / 'search_results.json'}")


if __name__ == "__main__":
    main()
