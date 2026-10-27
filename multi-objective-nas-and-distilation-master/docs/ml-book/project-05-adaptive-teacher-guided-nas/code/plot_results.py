import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot saved final-project candidate metrics")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("candidate_tradeoff.png"))
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    candidates = result["candidates"]
    if not candidates:
        raise ValueError("results contain no candidates")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in candidates:
        axes[0].scatter(row["latency_ms"], row["validation_accuracy"])
        axes[0].annotate(str(row["candidate_id"]),
                         (row["latency_ms"], row["validation_accuracy"]))
        axes[1].scatter(row["complexity_proxy"], row["alpha_ce"])
        axes[1].annotate(str(row["candidate_id"]),
                         (row["complexity_proxy"], row["alpha_ce"]))
    axes[0].set(title="Measured local validation trade-off", xlabel="Latency (ms; lower better)",
                ylabel="Validation accuracy (higher better)")
    axes[1].set(title="Adaptive CE weight by capacity proxy",
                xlabel="Depth × width² proxy", ylabel="CE weight alpha")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"saved plot to {args.output}; energy is {result['protocol']['energy']}")


if __name__ == "__main__":
    main()
