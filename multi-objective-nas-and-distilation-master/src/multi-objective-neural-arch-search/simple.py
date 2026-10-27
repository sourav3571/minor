import json
from pathlib import Path

DATA_FILE = Path(__file__).with_name("model_evaluations.json")

OBJECTIVES = {
    "accuracy": {
        "key": "accuracy_pct",
        "label": "accuracy (%)",
        "direction": "max",
    },
    "latency": {
        "key": "latency_ms",
        "label": "latency (ms; lower is better)",
        "direction": "min",
    },
    "energy": {
        "key": "energy_mj_per_image",
        "label": "energy (mJ per image; lower is better)",
        "direction": "min",
    },
    "size": {
        "key": "model_size_mb",
        "label": "model file size (MB; lower is better)",
        "direction": "min",
    },
    "ram": {
        "key": "peak_ram_mb",
        "label": "peak RAM (MB; lower is better)",
        "direction": "min",
    },
}

ALIASES = {
    "acc": "accuracy",
    "accuracy_pct": "accuracy",
    "speed": "latency",
    "latency_ms": "latency",
    "power": "energy",
    "energy_mj_per_image": "energy",
    "model_size": "size",
    "model_size_mb": "size",
    "memory": "ram",
    "peak_ram_mb": "ram",
}


def load_models():
    with DATA_FILE.open(encoding="utf-8") as file:
        data = json.load(file)
    return data["metadata"], data["models"]


def normalize_objective(name):
    name = name.strip().lower().replace(" ", "_")
    name = ALIASES.get(name, name)
    if name not in OBJECTIVES:
        choices = ", ".join(OBJECTIVES)
        raise ValueError(f"Unknown objective '{name}'. Choose: {choices}.")
    return name


def normalized_quality(model, objective, models):
    details = OBJECTIVES[objective]
    key = details["key"]
    values = [item[key] for item in models]
    low, high = min(values), max(values)
    if high == low:
        return 1.0

    value = model[key]
    if details["direction"] == "max":
        return (value - low) / (high - low)
    return (high - value) / (high - low)


def normalized_violation(model, objective, target):
    details = OBJECTIVES[objective]
    value = model[details["key"]]
    scale = max(abs(target), 1.0e-12)

    if details["direction"] == "max":
        miss = max(0.0, target - value)
    else:
        miss = max(0.0, value - target)
    return miss / scale


def meets_target(model, objective, target):
    details = OBJECTIVES[objective]
    value = model[details["key"]]
    if details["direction"] == "max":
        return value >= target
    return value <= target


def describe(model):
    channels = ", ".join(str(value) for value in model["channels"])
    print(f"\nRecommended model: {model['model_id']}")
    print(
        f"Design: depth={model['depth']}, channels=[{channels}], "
        f"kernel={model['kernel']}x{model['kernel']}"
    )
    print(f"Accuracy: {model['accuracy_pct']:.1f}%")
    print(f"Latency: {model['latency_ms']:.1f} ms per image")
    print(f"Energy: {model['energy_mj_per_image']:.1f} mJ per image")
    print(f"Model file: {model['model_size_mb']:.2f} MB")
    print(f"Peak RAM: {model['peak_ram_mb']:.1f} MB")


def main():
    metadata, models = load_models()
    print("Simple model selector")
    print("Data warning:", metadata["warning"])
    print("Objectives: accuracy, latency, energy, size, ram")
    print("Enter one or more names, separated by commas.")

    try:
        raw_names = input("Objectives: ").split(",")
        objectives = list(dict.fromkeys(normalize_objective(name) for name in raw_names))
        if not objectives:
            raise ValueError("Choose at least one objective.")

        targets = {}
        for objective in objectives:
            details = OBJECTIVES[objective]
            if details["direction"] == "max":
                prompt = f"Minimum desired {details['label']}: "
            else:
                prompt = f"Maximum allowed {details['label']}: "
            targets[objective] = float(input(prompt))
    except ValueError as error:
        print(f"Input error: {error}")
        return

    eligible = [
        model
        for model in models
        if all(meets_target(model, name, targets[name]) for name in objectives)
    ]

    if eligible:
        print(f"\n{len(eligible)} model(s) meet every target.")
        print("Selected objectives have equal importance.")
        ranked = sorted(
            eligible,
            key=lambda model: sum(
                normalized_quality(model, name, models) for name in objectives
            ) / len(objectives),
            reverse=True,
        )
        describe(ranked[0])
        return

    print("\nNo model meets every target.")
    print("Showing the model with the smallest total target miss.")
    ranked = sorted(
        models,
        key=lambda model: sum(
            normalized_violation(model, name, targets[name]) for name in objectives
        ),
    )
    describe(ranked[0])


if __name__ == "__main__":
    main()
