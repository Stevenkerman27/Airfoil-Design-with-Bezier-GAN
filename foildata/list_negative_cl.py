import argparse
from collections import defaultdict
from pathlib import Path

import torch


DATA_PATH = Path(__file__).resolve().parent.parent / "model" / "airfoil_dataset.pt"


def collect_samples(threshold):
    if threshold >= 0:
        raise ValueError("threshold must be negative")

    data = torch.load(DATA_PATH, weights_only=True)
    samples = defaultdict(list)
    for item in data:
        alpha, reynolds, cl, _ = item["y"].tolist()
        if cl < threshold:
            samples[item["foil_id"]].append((cl, alpha, reynolds))
    return samples


def parse_threshold():
    parser = argparse.ArgumentParser(
        description="List airfoils with samples whose Cl is below a threshold."
    )
    parser.add_argument(
        "threshold",
        nargs="?",
        type=float,
        help="strict filter: Cl < threshold; must be negative",
    )
    args = parser.parse_args()
    if args.threshold is not None:
        return args.threshold
    return float(input("Enter a negative Cl threshold: "))


def main():
    threshold = parse_threshold()
    samples = collect_samples(threshold)

    print(f"\nFilter: Cl < {threshold:g}")
    print(f"Airfoils: {len(samples)}")
    print(f"Matching samples: {sum(len(values) for values in samples.values())}\n")
    print(f"{'Airfoil':<28} {'Count':>7} {'Min Cl':>12} {'Alpha':>10} {'Re':>12}")
    print("-" * 75)
    for foil_id in sorted(samples):
        values = samples[foil_id]
        min_cl, alpha, reynolds = min(values, key=lambda value: value[0])
        print(f"{foil_id:<28} {len(values):>7} {min_cl:>12.6f} {alpha:>10.4f} {reynolds:>12.0f}")


if __name__ == "__main__":
    main()
