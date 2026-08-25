#!/usr/bin/env python3
"""Plot the shear-wave velocity at y = Y/4 as a function of time."""

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def timestep(path: Path) -> int:
    """Extract the timestep from a name such as distribution_40.csv."""
    match = re.search(r"(\d+)(?=\.csv$)", path.name)
    if match is None:
        raise ValueError(f"Cannot determine timestep from {path}")
    return int(match.group(1))


def velocity_at_quarter_height(path: Path) -> float:
    """Return the x-averaged u_x value at y = floor(Y/4)."""
    values: dict[tuple[int, int], float] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x_index", "y_index", "u_x"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        for row in reader:
            position = (int(row["x_index"]), int(row["y_index"]))
            velocity = float(row["u_x"])
            previous = values.setdefault(position, velocity)
            if not np.isclose(previous, velocity):
                raise ValueError(f"Inconsistent u_x values at {position} in {path}")

    if not values:
        raise ValueError(f"{path} contains no velocity data")

    max_y = max(y for _, y in values)
    grid_height = max_y + 1
    target_y = grid_height // 4
    samples = [velocity for (_, y), velocity in values.items() if y == target_y]
    if not samples:
        raise ValueError(f"{path} contains no samples at y={target_y}")
    return float(np.mean(samples))


def plot_decay(input_pattern: str, output_path: str) -> None:
    """Plot u_x(y=Y/4) over time for all CSV files matching input_pattern."""
    files = sorted((Path(path) for path in glob.glob(input_pattern)), key=timestep)
    if not files:
        raise FileNotFoundError(f"No CSV files match {input_pattern!r}")

    times = np.array([timestep(path) for path in files])
    velocities = np.array([velocity_at_quarter_height(path) for path in files])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(times, velocities, marker="o", markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"Mean $u_x$ at $y=Y/4$")
    ax.set_title("Shear-wave velocity decay")
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output} ({len(files)} timesteps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="?",
        default="out/data/distribution_*.csv",
        help="CSV glob (default: out/data/distribution_*.csv)",
    )
    parser.add_argument("--output", default="out/velocity_decay.png")
    args = parser.parse_args()
    plot_decay(args.inputs, args.output)


if __name__ == "__main__":
    main()
