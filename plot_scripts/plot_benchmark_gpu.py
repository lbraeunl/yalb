#!/usr/bin/env python3
"""Plot GPU and serial-CPU throughput against the number of lattice points."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_benchmark(path: Path) -> dict[int, list[float]]:
    """Read lattice-point and MLURPS values, grouped by lattice size."""
    measurements: dict[int, list[float]] = defaultdict(list)

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"lattice_points", "mlurps"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )

        for row in reader:
            measurements[int(row["lattice_points"])].append(float(row["mlurps"]))

    if not measurements:
        raise ValueError(f"{path} contains no benchmark data")
    return dict(measurements)


def mean_throughput(path: Path) -> tuple[list[int], list[float]]:
    """Return sorted lattice sizes and mean MLURPS values."""
    measurements = read_benchmark(path)
    lattice_points = sorted(measurements)
    means = [
        sum(measurements[size]) / len(measurements[size])
        for size in lattice_points
    ]
    return lattice_points, means


def plot_benchmark(gpu_path: Path, cpu_path: Path, output_path: Path) -> None:
    """Plot mean GPU and serial-CPU MLURPS for each lattice size."""
    gpu_points, gpu_means = mean_throughput(gpu_path)
    cpu_points, cpu_means = mean_throughput(cpu_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(constrained_layout=True)
    gpu_color = "tab:blue"
    cpu_color = "tab:orange"
    ax.plot(
        gpu_points,
        gpu_means,
        color=gpu_color,
        marker="o",
        markersize=4,
    )
    ax.plot(
        cpu_points,
        cpu_means,
        color=cpu_color,
        marker="o",
        markersize=4,
    )
    ax.annotate(
        "GPU",
        (gpu_points[-1], gpu_means[-1]),
        xytext=(-8, -10),
        textcoords="offset points",
        color=gpu_color,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.annotate(
        "Serial CPU",
        (cpu_points[-1], cpu_means[-1]),
        xytext=(6, -10),
        textcoords="offset points",
        color=cpu_color,
        fontweight="bold",
    )
    ax.set_xlabel("Lattice points")
    ax.set_ylabel("MLUPS")
    ax.set_xscale("log", base=2)
    all_points = gpu_points + cpu_points
    ax.set_xlim(min(all_points) / 1.1, max(all_points) * 1.5)
    ax.set_title("Benchmark performance")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(
        f"Wrote {output_path} "
        f"({len(gpu_points)} GPU and {len(cpu_points)} CPU lattice sizes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("out/benchmark_gpu.csv"),
        help="GPU benchmark CSV (default: out/benchmark_gpu.csv)",
    )
    parser.add_argument(
        "--cpu",
        type=Path,
        default=Path("out/benchmark_cpu_serial.csv"),
        help="serial CPU benchmark CSV (default: out/benchmark_cpu_serial.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/benchmark_gpu.png"),
        help="output image (default: out/benchmark_gpu.png)",
    )
    args = parser.parse_args()
    plot_benchmark(args.input, args.cpu, args.output)


if __name__ == "__main__":
    main()
