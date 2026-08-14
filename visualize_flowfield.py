#!/usr/bin/env python3
"""Create a velocity-magnitude and streamline animation from CSV snapshots."""

import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


def numeric_suffix(path: Path) -> tuple[int, str]:
    """Sort distribution_2.csv before distribution_10.csv."""
    match = re.search(r"(\d+)(?=\.csv$)", path.name)
    return (int(match.group(1)) if match else -1, str(path))


def resolve_inputs(pattern: str) -> list[Path]:
    """Resolve either a directory or a glob into numerically sorted CSV files."""
    candidate = Path(pattern)
    files = (
        list(candidate.glob("*.csv"))
        if candidate.is_dir()
        else [Path(path) for path in glob.glob(pattern)]
    )
    files.sort(key=numeric_suffix)
    if not files:
        raise FileNotFoundError(f"No CSV files match {pattern!r}")
    return files


def read_velocity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one velocity vector per lattice cell from a distribution CSV."""
    values: dict[tuple[int, int], tuple[float, float]] = {}

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x_index", "y_index", "u_x", "u_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            columns = ", ".join(sorted(required))
            raise ValueError(f"{path} must contain columns: {columns}")

        for row in reader:
            position = (int(row["x_index"]), int(row["y_index"]))
            velocity = (float(row["u_x"]), float(row["u_y"]))
            previous = values.setdefault(position, velocity)
            if not np.allclose(previous, velocity):
                raise ValueError(
                    f"Inconsistent velocity values at {position} in {path}"
                )

    if not values:
        raise ValueError(f"{path} contains no data rows")

    x_indices = [position[0] for position in values]
    y_indices = [position[1] for position in values]
    if min(x_indices) < 0 or min(y_indices) < 0:
        raise ValueError(f"{path} contains negative lattice indices")

    shape = (max(y_indices) + 1, max(x_indices) + 1)
    u_x = np.full(shape, np.nan)
    u_y = np.full(shape, np.nan)
    for (x_index, y_index), (velocity_x, velocity_y) in values.items():
        u_x[y_index, x_index] = velocity_x
        u_y[y_index, x_index] = velocity_y

    if np.isnan(u_x).any() or np.isnan(u_y).any():
        raise ValueError(f"{path} does not contain a complete rectangular grid")
    if not np.isfinite(u_x).all() or not np.isfinite(u_y).all():
        raise ValueError(f"{path} contains non-finite velocity values")

    return u_x, u_y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="?",
        default="out/data/distribution_*.csv",
        help="CSV glob or directory (default: out/data/distribution_*.csv)",
    )
    parser.add_argument(
        "--output",
        default="out/flowfield_animation.gif",
        help="Output GIF path (default: out/flowfield_animation.gif)",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0, help="Frames per second (default: 5)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.2,
        help="Streamline density (default: 1.2)",
    )
    parser.add_argument(
        "--steady",
        action="store_true",
        help="Render only the last matching CSV snapshot",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.density <= 0:
        parser.error("--density must be positive")

    files = resolve_inputs(args.inputs)
    if args.steady:
        files = files[-1:]
    fields = [read_velocity(path) for path in files]
    shape = fields[0][0].shape
    if any(u_x.shape != shape or u_y.shape != shape for u_x, u_y in fields):
        raise ValueError("All CSV files must cover the same x/y grid")

    speeds = [np.hypot(u_x, u_y) for u_x, u_y in fields]
    maximum_speed = max(float(np.max(speed)) for speed in speeds)
    if maximum_speed == 0.0:
        maximum_speed = 1.0

    x_coordinates = np.arange(shape[1])
    y_coordinates = np.arange(shape[0])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(constrained_layout=True)
    color_scale = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=maximum_speed), cmap="viridis"
    )
    fig.colorbar(color_scale, ax=ax, label=r"Speed $|\mathbf{u}|$")

    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, str(output), dpi=120):
        for number, (path, (u_x, u_y), speed) in enumerate(
            zip(files, fields, speeds), start=1
        ):
            ax.clear()
            ax.imshow(
                speed,
                origin="lower",
                extent=(0, shape[1] - 1, 0, shape[0] - 1),
                cmap="viridis",
                vmin=0.0,
                vmax=maximum_speed,
                interpolation="nearest",
                aspect="equal",
            )
            ax.streamplot(
                x_coordinates,
                y_coordinates,
                u_x,
                u_y,
                color="white",
                density=args.density,
                linewidth=0.8,
                arrowsize=0.8,
            )
            ax.set(
                xlabel="x index",
                ylabel="y index",
                title=f"Velocity field: {path.name} ({number}/{len(files)})",
                xlim=(0, shape[1] - 1),
                ylim=(0, shape[0] - 1),
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {output} ({len(files)} frames)")


if __name__ == "__main__":
    main()
