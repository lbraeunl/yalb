#!/usr/bin/env python3
"""Animate the x-velocity profile as a function of the y-coordinate."""

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


def numeric_suffix(path: str) -> tuple[int, str]:
    """Sort distribution_2.csv before distribution_10.csv."""
    match = re.search(r"(\d+)(?=\.csv$)", Path(path).name)
    return (int(match.group(1)) if match else -1, path)


def resolve_inputs(pattern: str) -> list[Path]:
    candidate = Path(pattern)
    files = (
        sorted(candidate.glob("*.csv"), key=lambda p: numeric_suffix(str(p)))
        if candidate.is_dir()
        else sorted((Path(p) for p in glob.glob(pattern)), key=lambda p: numeric_suffix(str(p)))
    )
    if not files:
        raise FileNotFoundError(f"No CSV files match {pattern!r}")
    return files


def read_velocity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one velocity vector per lattice cell, ignoring repeated populations."""
    values: dict[tuple[int, int], tuple[float, float]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x_index", "y_index", "u_x", "u_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        for row in reader:
            position = (int(row["x_index"]), int(row["y_index"]))
            velocity = (float(row["u_x"]), float(row["u_y"]))
            previous = values.setdefault(position, velocity)
            if not np.allclose(previous, velocity):
                raise ValueError(f"Inconsistent velocity values at {position} in {path}")

    if not values:
        raise ValueError(f"{path} contains no data rows")

    max_x = max(x for x, _ in values)
    max_y = max(y for _, y in values)
    u_x = np.full((max_y + 1, max_x + 1), np.nan)
    u_y = np.full_like(u_x, np.nan)
    for (x, y), (velocity_x, velocity_y) in values.items():
        u_x[y, x] = velocity_x
        u_y[y, x] = velocity_y
    return u_x, u_y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="?",
        default="out/data/distribution_*.csv",
        help="CSV glob or directory (default: out/data/distribution_*.csv)",
    )
    parser.add_argument("--output", default="out/velocity_animation.gif")
    parser.add_argument("--fps", type=float, default=5, help="Frames per second (default: 5)")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    files = resolve_inputs(args.inputs)
    fields = [read_velocity(path) for path in files]
    shape = fields[0][0].shape
    if any(u_x.shape != shape or u_y.shape != shape for u_x, u_y in fields):
        raise ValueError("All CSV files must cover the same x/y grid")

    profiles = [np.nanmean(u_x, axis=1) for u_x, _ in fields]
    max_abs_velocity = max(np.nanmax(np.abs(profile)) for profile in profiles)
    if not np.isfinite(max_abs_velocity) or max_abs_velocity == 0:
        max_abs_velocity = 1.0
    velocity_limit = 1.05 * max_abs_velocity
    y_coordinates = np.arange(shape[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(constrained_layout=True)
    (line,) = ax.plot(y_coordinates, profiles[0], marker="o", markersize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(0, shape[0] - 1)
    ax.set_ylim(-velocity_limit, velocity_limit)
    ax.set_xlabel("y index")
    ax.set_ylabel(r"Mean x-velocity $\langle u_x \rangle_x$")
    ax.grid(alpha=0.25)

    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, str(output), dpi=100):
        for number, (path, profile) in enumerate(zip(files, profiles), start=1):
            line.set_ydata(profile)
            ax.set_title(f"Shear-wave velocity: {path.name} ({number}/{len(files)})")
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {output} ({len(files)} frames)")


if __name__ == "__main__":
    main()
