#!/usr/bin/env python3
"""Create a density heatmap animation from distribution CSV output files."""

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


def read_density(path: Path) -> np.ndarray:
    """Read one density value per (x_index, y_index), ignoring populations."""
    values: dict[tuple[int, int], float] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x_index", "y_index", "density_value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            position = (int(row["x_index"]), int(row["y_index"]))
            density = float(row["density_value"])
            previous = values.setdefault(position, density)
            if not np.isclose(previous, density):
                raise ValueError(f"Inconsistent density values at {position} in {path}")

    if not values:
        raise ValueError(f"{path} contains no data rows")

    max_x = max(x for x, _ in values)
    max_y = max(y for _, y in values)
    field = np.full((max_y + 1, max_x + 1), np.nan)
    for (x, y), density in values.items():
        field[y, x] = density
    return field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="?",
        default="out/distribution_*.csv",
        help="CSV glob or directory (default: out/distribution_*.csv)",
    )
    parser.add_argument("--output", default="out/density_animation.gif")
    parser.add_argument("--fps", type=float, default=5, help="Frames per second (default: 10)")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    files = resolve_inputs(args.inputs)
    fields = [read_density(path) for path in files]
    shape = fields[0].shape
    if any(field.shape != shape for field in fields[1:]):
        raise ValueError("All CSV files must cover the same x/y grid")

    vmin = min(np.nanmin(field) for field in fields)
    vmax = max(np.nanmax(field) for field in fields)
    if vmin == vmax:
        vmax = vmin + 1e-12

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(constrained_layout=True)
    image = ax.imshow(fields[0], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(image, ax=ax, label="Density")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, str(output), dpi=100):
        for number, (path, field) in enumerate(zip(files, fields), start=1):
            image.set_data(field)
            ax.set_title(f"Density: {path.name} ({number}/{len(files)})")
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {output} ({len(files)} frames)")


if __name__ == "__main__":
    main()
