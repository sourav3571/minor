import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self, channels=16, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(channels * 2 * 7 * 7, classes)

    def forward(self, images):
        features = self.features(images)
        assert features.shape[1:] == (self.classifier.in_features // 49, 7, 7)
        return self.classifier(features.flatten(start_dim=1))


class SmallMLP(nn.Module):
    def __init__(self, hidden=128, classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(), nn.Linear(hidden, classes)
        )

    def forward(self, images):
        return self.layers(images)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(dataset_name, root, batch_size, seed, train_limit, test_limit, workers):
    transform = transforms.ToTensor()
    dataset_type = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    full_train = dataset_type(root=root, train=True, download=True, transform=transform)
    test_set = dataset_type(root=root, train=False, download=True, transform=transform)

    if train_limit:
        full_train = Subset(full_train, range(min(train_limit, len(full_train))))
    if test_limit:
        test_set = Subset(test_set, range(min(test_limit, len(test_set))))

    validation_size = max(1, int(0.2 * len(full_train)))
    training_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(seed)
    training_set, validation_set = random_split(
        full_train, [training_size, validation_size], generator=generator
    )
    loader_generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(training_set, shuffle=True, generator=loader_generator, **common)
    validation_loader = DataLoader(validation_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, validation_loader, test_loader, training_size, validation_size, len(test_set)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    predictions = []
    targets = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            assert logits.shape == (images.shape[0], 10)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            count += batch_size
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return total_loss / count, correct / count, predictions, targets


def confusion_matrix(targets, predictions, classes=10):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def classification_metrics(matrix):
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0)
    actual_count = matrix.sum(axis=1)
    precision = np.divide(true_positive, predicted_count,
                          out=np.zeros_like(true_positive), where=predicted_count != 0)
    recall = np.divide(true_positive, actual_count,
                       out=np.zeros_like(true_positive), where=actual_count != 0)
    return {"macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean())}


def main():
    parser = argparse.ArgumentParser(description="Train a small MNIST/Fashion-MNIST classifier")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/project1"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.channels < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, channels must be positive and learning-rate > 0")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader, train_n, val_n, test_n = make_loaders(
        args.dataset, args.data_dir, args.batch_size, args.seed, args.train_limit,
        args.test_limit, args.workers
    )
    sample_images, sample_labels = next(iter(train_loader))
    print(f"device={device}; batch={tuple(sample_images.shape)}; dtype={sample_images.dtype}; "
          f"label_batch={tuple(sample_labels.shape)}; first_label={sample_labels[0].item()}")
    assert sample_images.ndim == 4 and sample_images.shape[1:] == (1, 28, 28)

    model = SmallCNN(args.channels) if args.model == "cnn" else SmallMLP()
    model = model.to(device)
    with torch.no_grad():
        shape_check = model(sample_images[:2].to(device))
    assert shape_check.shape == (2, 10)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_accuracy = -1.0
    history = []
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        validation_loss, validation_accuracy, _, _ = run_epoch(
            model, validation_loader, loss_fn, device
        )
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy})
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
              f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.3f}")
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save({"model": model.state_dict(), "model_type": args.model,
                        "channels": args.channels, "classes": 10, "seed": args.seed,
                        "dataset": args.dataset}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded = SmallCNN(checkpoint["channels"]) if checkpoint["model_type"] == "cnn" else SmallMLP()
    reloaded.load_state_dict(checkpoint["model"])
    reloaded.to(device)
    test_loss, test_accuracy, predictions, targets = run_epoch(
        reloaded, test_loader, loss_fn, device
    )
    matrix = confusion_matrix(targets, predictions)
    metrics = classification_metrics(matrix)
    print(f"FINAL TEST (used once after selection): n={test_n} loss={test_loss:.4f} "
          f"accuracy={test_accuracy:.3f}")
    print(f"parameters={parameter_count}; best_validation_accuracy={best_validation_accuracy:.3f}")

    with torch.no_grad():
        example_images, example_labels = next(iter(test_loader))
        example_logits = reloaded(example_images[:8].to(device))
        example_probabilities = example_logits.softmax(dim=1).cpu()
    examples = [{"true": int(example_labels[i]),
                 "predicted": int(example_probabilities[i].argmax()),
                 "probabilities": [round(float(value), 4) for value in example_probabilities[i]]}
                for i in range(min(8, len(example_labels)))]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axes[0].plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot([row["epoch"] for row in history], [row["train_accuracy"] for row in history], label="train")
    axes[1].plot([row["epoch"] for row in history], [row["validation_accuracy"] for row in history], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Fraction correct")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title="Test confusion matrix", xlabel="Predicted class", ylabel="True class")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    record = {"args": {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()},
              "device": str(device), "torch_version": torch.__version__,
              "train_examples": train_n, "validation_examples": val_n, "test_examples": test_n,
              "parameters": parameter_count, "history": history,
              "best_validation_accuracy": best_validation_accuracy,
              "test_loss": test_loss, "test_accuracy": test_accuracy,
              "test_macro_precision": metrics["macro_precision"],
              "test_macro_recall": metrics["macro_recall"],
              "test_confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
              "predictions": examples}
    (args.output_dir / "run.json").write_text(json.dumps(record, indent=2))
    print(f"saved={checkpoint_path}; curves={args.output_dir / 'learning_curves.png'}")
    print(f"first prediction example={examples[0]}")


if __name__ == "__main__":
    main()
