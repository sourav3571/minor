import io
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ILLUSTRATIVE = [
    {"name": "A", "accuracy": 0.90, "latency_ms": 10.0, "size_mb": 5.0},
    {"name": "B", "accuracy": 0.88, "latency_ms": 8.0, "size_mb": 4.0},
    {"name": "C", "accuracy": 0.92, "latency_ms": 15.0, "size_mb": 6.0},
    {"name": "D", "accuracy": 0.85, "latency_ms": 12.0, "size_mb": 7.0},
]


def is_better_or_equal(left, right, directions):
    return all(a >= b if direction == "max" else a <= b
               for a, b, direction in zip(left, right, directions))


def dominates(left, right, directions):
    no_worse = is_better_or_equal(left, right, directions)
    strictly_better = any(a > b if direction == "max" else a < b
                          for a, b, direction in zip(left, right, directions))
    return no_worse and strictly_better


def pareto_indices(rows, objectives, directions, constraints=None):
    if len(objectives) != len(directions):
        raise ValueError("each objective needs a direction")
    constraints = constraints or {}
    feasible = [index for index, row in enumerate(rows)
                if all(row[key] <= limit for key, limit in constraints.items())]
    frontier = []
    for candidate_index in feasible:
        candidate = [rows[candidate_index][key] for key in objectives]
        if not any(
            other_index != candidate_index and dominates(
                [rows[other_index][key] for key in objectives], candidate, directions
            )
            for other_index in feasible
        ):
            frontier.append(candidate_index)
    return frontier


class CandidateCNN(nn.Module):
    def __init__(self, channels, depth, kernel):
        super().__init__()
        blocks = []
        incoming = 1
        for _ in range(depth):
            blocks.extend([nn.Conv2d(incoming, channels, kernel, padding="same"),
                           nn.ReLU(), nn.MaxPool2d(2)])
            incoming = channels
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            output = self.features(torch.zeros(1, 1, 28, 28))
        self.classifier = nn.Linear(int(np.prod(output.shape[1:])), 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def measure_latency(model, device, repeats=100):
    model.eval()
    sample = torch.rand(1, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(20):
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


def benchmark_untrained_candidates(device):
    configurations = [
        {"name": "small", "channels": 8, "depth": 1, "kernel": 3},
        {"name": "medium", "channels": 16, "depth": 2, "kernel": 3},
        {"name": "wide", "channels": 24, "depth": 2, "kernel": 5},
    ]
    rows = []
    for config in configurations:
        model = CandidateCNN(config["channels"], config["depth"], config["kernel"]).to(device)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        row = {**config, "parameters": sum(p.numel() for p in model.parameters()),
               "state_dict_bytes": buffer.tell(),
               "latency": measure_latency(model, device)}
        rows.append(row)
    return rows


def main():
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    output_dir = Path("artifacts/project4")
    output_dir.mkdir(parents=True, exist_ok=True)

    illustrative_frontier = pareto_indices(
        ILLUSTRATIVE, ["accuracy", "latency_ms", "size_mb"], ["max", "min", "min"]
    )
    print("Illustrative Pareto candidates (NOT measurements):",
          [ILLUSTRATIVE[index]["name"] for index in illustrative_frontier])
    figure, axis = plt.subplots(figsize=(7, 5))
    for index, row in enumerate(ILLUSTRATIVE):
        axis.scatter(row["latency_ms"], row["accuracy"],
                     marker="o" if index in illustrative_frontier else "x")
        axis.annotate(row["name"], (row["latency_ms"], row["accuracy"]))
    axis.set(title="Illustrative accuracy-latency trade-off (not experimental data)",
             xlabel="Latency (illustrative ms; lower is better)",
             ylabel="Accuracy (illustrative; higher is better)")
    figure.tight_layout()
    figure.savefig(output_dir / "illustrative_pareto.png", dpi=150)
    plt.close(figure)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measured = benchmark_untrained_candidates(device)
    print(f"Local device={device}; candidate weights are UNTRAINED, so no accuracy is claimed")
    for row in measured:
        print(f"{row['name']}: params={row['parameters']} bytes={row['state_dict_bytes']} "
              f"median_ms={row['latency']['median_ms']:.4f} p95_ms={row['latency']['p95_ms']:.4f}")
    result = {"illustrative_only": ILLUSTRATIVE,
              "illustrative_frontier_indices": illustrative_frontier,
              "local_untrained_benchmark": measured,
              "protocol": {"device": str(device), "batch": 1,
                           "input_shape": [1, 1, 28, 28], "warmup": 20,
                           "latency_repeats": 100,
                           "scope": "model forward only; synthetic input; untrained models",
                           "energy": "not measured"}}
    (output_dir / "measurements.json").write_text(json.dumps(result, indent=2))
    print(f"saved {output_dir / 'measurements.json'} and illustrative plot")


if __name__ == "__main__":
    main()
