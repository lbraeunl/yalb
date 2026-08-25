#!/usr/bin/env python3
"""Plot the CPU-to-GPU mean time-to-solution ratio."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MAX_LATTICE_POINTS = 65_536


def read_mean_times(path: Path) -> dict[int, float]:
    """Return the mean time to solution for each lattice size in a CSV file."""
    measurements: dict[int, list[float]] = defaultdict(list)

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"lattice_points", "time_to_solution_seconds"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )

        for row in reader:
            lattice_points = int(row["lattice_points"])
            if lattice_points <= MAX_LATTICE_POINTS:
                measurements[lattice_points].append(
                    float(row["time_to_solution_seconds"])
                )

    if not measurements:
        raise ValueError(
            f"{path} contains no benchmark data at or below "
            f"{MAX_LATTICE_POINTS} lattice points"
        )

    return {
        lattice_points: sum(times) / len(times)
        for lattice_points, times in measurements.items()
    }


def plot_time_ratio(gpu_path: Path, cpu_path: Path, output_path: Path) -> None:
    """Plot mean CPU time divided by mean GPU time for common lattice sizes."""
    gpu_times = read_mean_times(gpu_path)
    cpu_times = read_mean_times(cpu_path)
    lattice_points = sorted(gpu_times.keys() & cpu_times.keys())

    if not lattice_points:
        raise ValueError(
            "The CPU and GPU CSV files have no common lattice sizes at or below "
            f"{MAX_LATTICE_POINTS}"
        )

    ratios = [cpu_times[size] / gpu_times[size] for size in lattice_points]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(lattice_points, ratios, marker="o")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Lattice points")
    ax.set_ylabel("T(CPU) / T(GPU)")
    ax.set_xscale("log", base=2)
    ax.set_title("GPU speedup over Serial CPU")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path} ({len(lattice_points)} lattice sizes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path("out/benchmark_gpu.csv"),
        help="GPU benchmark CSV (default: out/benchmark_gpu.csv)",
    )
    parser.add_argument(
        "--cpu",
        type=Path,
        default=Path("out/benchmark_cpu_serial.csv"),
        help="CPU benchmark CSV (default: out/benchmark_cpu_serial.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/benchmark_time_ratio.png"),
        help="output image (default: out/benchmark_time_ratio.png)",
    )
    args = parser.parse_args()
    plot_time_ratio(args.gpu, args.cpu, args.output)


if __name__ == "__main__":
    main()
