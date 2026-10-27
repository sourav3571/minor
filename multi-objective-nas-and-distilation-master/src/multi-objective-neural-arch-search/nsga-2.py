import json
import math
from pathlib import Path

DATA_FILE = Path(__file__).with_name("model_evaluations.json")

# t- max
OBJECTIVES = [
    ("accuracy_pct", "accuracy (%)", True),
    ("latency_ms", "latency (ms)", False),
    ("energy_mj_per_image", "energy (mJ/image)", False),
    ("model_size_mb", "model file size (MB)", False),
    ("peak_ram_mb", "peak RAM (MB)", False),
]


def load_data():
    with DATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def objective_value(model, key, maximize):
    value = model[key]
    return -value if maximize else value


def dominates(first, second):
    first_values = [
        objective_value(first, key, maximize)
        for key, _label, maximize in OBJECTIVES
    ]
    second_values = [
        objective_value(second, key, maximize)
        for key, _label, maximize in OBJECTIVES
    ]
    no_worse = all(left <= right for left, right in zip(first_values, second_values))
    strictly_better = any(left < right for left, right in zip(first_values, second_values))
    return no_worse and strictly_better


def fast_non_dominated_sort(models):
    count = len(models)
    dominates_list = [[] for _ in models]
    dominated_by_count = [0] * count
    first_front = []

    for first_index, first in enumerate(models):
        for second_index, second in enumerate(models):
            if first_index == second_index:
                continue
            if dominates(first, second):
                dominates_list[first_index].append(second_index)
            elif dominates(second, first):
                dominated_by_count[first_index] += 1

        if dominated_by_count[first_index] == 0:
            first_front.append(first_index)

    fronts = [first_front]
    current_front = first_front
    while current_front:
        next_front = []
        for first_index in current_front:
            for second_index in dominates_list[first_index]:
                dominated_by_count[second_index] -= 1
                if dominated_by_count[second_index] == 0:
                    next_front.append(second_index)
        if next_front:
            fronts.append(next_front)
        current_front = next_front

    return fronts


def crowding_distances(front, models):
    distances = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: math.inf for index in front}

    for key, _label, _maximize in OBJECTIVES:
        ordered = sorted(front, key=lambda index: models[index][key])
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        low = models[ordered[0]][key]
        high = models[ordered[-1]][key]
        if high == low:
            continue

        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if not math.isinf(distances[index]):
                before = models[ordered[position - 1]][key]
                after = models[ordered[position + 1]][key]
                distances[index] += (after - before) / (high - low)

    return distances


def print_model(model, rank, crowding):
    channels = ", ".join(str(value) for value in model["channels"])
    crowding_text = "edge" if math.isinf(crowding) else f"{crowding:.3f}"
    print(
        f"{model['model_id']:>10} | rank {rank} | crowding {crowding_text:>5} | "
        f"depth {model['depth']} | channels [{channels}] | "
        f"kernel {model['kernel']}x{model['kernel']}"
    )
    print(
        f"           accuracy {model['accuracy_pct']:.1f}% | "
        f"latency {model['latency_ms']:.1f} ms | "
        f"energy {model['energy_mj_per_image']:.1f} mJ/image | "
        f"file {model['model_size_mb']:.2f} MB | "
        f"peak RAM {model['peak_ram_mb']:.1f} MB"
    )


def main():
    data = load_data()
    models = data["models"]
    fronts = fast_non_dominated_sort(models)

    print("NSGA-II multi-objective results")
    print("Data warning:", data["metadata"]["warning"])
    print("Objectives: maximize accuracy; minimize latency, energy, size, and RAM.")
    print("Model size and peak RAM are shown as separate memory measures.\n")

    ranks = {}
    crowding = {}
    for rank, front in enumerate(fronts):
        front_distances = crowding_distances(front, models)
        for index in front:
            ranks[index] = rank
            crowding[index] = front_distances[index]

    pareto_front = fronts[0] if fronts else []
    pareto_front = sorted(
        pareto_front,
        key=lambda index: crowding[index],
        reverse=True,
    )

    print(f"Pareto-front candidates: {len(pareto_front)} of {len(models)}\n")
    print("A Pareto candidate is not beaten on every objective by another candidate.\n")
    for index in pareto_front:
        print_model(models[index], ranks[index], crowding[index])

    print("\nOther candidates, grouped by NSGA-II front:")
    for index, model in enumerate(models):
        if ranks[index] > 0:
            print_model(model, ranks[index], crowding[index])

    print(
        "\nThis selects among measured records. A full NAS loop must generate, "
        "train, and benchmark new candidates."
    )


if __name__ == "__main__":
    main()
