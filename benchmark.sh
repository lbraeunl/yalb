#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
preset="${BENCHMARK_PRESET:-cuda-release}"
build_dir="${BENCHMARK_BUILD_DIR:-${project_dir}/build-benchmark-${preset}}"
output_file="${BENCHMARK_OUTPUT:-${project_dir}/out/benchmark_shear_wave_cpu.csv}"
steps="${BENCHMARK_STEPS:-8000}"
repetitions="${BENCHMARK_REPETITIONS:-2}"
launcher="${BENCHMARK_LAUNCHER:-mpirun}"
jobs="${BENCHMARK_BUILD_JOBS:-2}"

if (($# > 0)); then
    sides=("$@")
else
    sides=(16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072)
fi

for value in "$steps" "$repetitions" "$jobs" "${sides[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Benchmark values must be positive integers; got: $value" >&2
        exit 1
    fi
done

mkdir -p "$(dirname -- "$output_file")"
if [[ "${BENCHMARK_APPEND:-0}" != "1" || ! -s "$output_file" ]]; then
    echo "side_length,lattice_points,steps,repetition,time_to_solution_seconds,mlurps" > "$output_file"
fi

for side in "${sides[@]}"; do
    echo "Configuring ${side} x ${side} lattice..."
    cmake --preset "$preset" \
        -B "$build_dir" \
        -DBUILD_TESTING=OFF \
        -DYALB_LATTICE_SIDE="$side" \
        -DYALB_NUM_STEPS="$steps" \
        -DYALB_DISABLE_FIELD_OUTPUT=ON
    cmake --build "$build_dir" --target main --parallel "$jobs"

    executable="${build_dir}/executables/main"
    for ((repetition = 1; repetition <= repetitions; ++repetition)); do
        echo "Running side=${side}, repetition=${repetition}/${repetitions}..."

        if [[ "$launcher" == "none" ]]; then
            command=("$executable")
        else
            command=("$launcher" -n 1 "$executable")
        fi

        if ! run_output="$("${command[@]}" 2>&1)"; then
            echo "$run_output" >&2
            exit 1
        fi
        echo "$run_output"

        if ! grep -q '^Running single-grid solver' <<< "$run_output"; then
            echo "Benchmark requires USE_DOMAIN_DECOMPOSITION = false" >&2
            exit 1
        fi

        elapsed_seconds="$(awk '/^Time-to-solution:/ {print $2}' <<< "$run_output")"
        throughput="$(awk '/^MLURPS:/ {print $2}' <<< "$run_output")"
        if [[ -z "$elapsed_seconds" || -z "$throughput" ]]; then
            echo "Could not parse performance output" >&2
            exit 1
        fi

        printf '%d,%d,%d,%d,%s,%s\n' \
            "$side" "$((side * side))" "$steps" "$repetition" \
            "$elapsed_seconds" "$throughput" >> "$output_file"
    done
done

echo "Benchmark results written to ${output_file}"
