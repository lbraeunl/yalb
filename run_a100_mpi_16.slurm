#!/bin/bash
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=yalb-a100-mpi-16
#SBATCH --output=slurm-a100-mpi-16-%j.out
#SBATCH --export=ALL

set -euo pipefail

module purge
module load compiler/gnu mpi/openmpi
module load toolkit/nvidia-hpc-sdk/25.1

executable="${SLURM_SUBMIT_DIR}/build-a100/executables/main"
run_directory="${SLURM_SUBMIT_DIR}/results/a100-mpi-16/${SLURM_JOB_ID}"

if [[ ! -x "${executable}" ]]; then
    echo "Executable not found: ${executable}" >&2
    echo "Build it first with: cmake --preset a100-release && cmake --build --preset a100-release -j" >&2
    exit 1
fi

mkdir -p "${run_directory}"
cd "${run_directory}"

echo "Running ${SLURM_NTASKS} MPI rank(s) on ${SLURM_JOB_NUM_NODES} node(s)."
echo "Results will be stored in ${run_directory}/out/data."
nvidia-smi --query-gpu=index,name,uuid --format=csv

time mpirun "${executable}"
