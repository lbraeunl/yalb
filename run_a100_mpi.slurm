#!/bin/bash
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=yalb-a100-mpi
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Use the same modules that were loaded when configuring and compiling.
module purge
module load compiler/gnu mpi/openmpi
module load toolkit/nvidia-hpc-sdk/25.1

executable="${SLURM_SUBMIT_DIR}/build-a100/executables/main"

if [[ ! -x "${executable}" ]]; then
    echo "Executable not found: ${executable}" >&2
    echo "Build it first with: cmake --preset a100-release && cmake --build --preset a100-release -j" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

echo "Running ${SLURM_NTASKS:-unknown} MPI rank(s) on ${SLURM_JOB_NUM_NODES} node(s)."
echo "MPI ranks per node: ${SLURM_NTASKS_PER_NODE:-unknown}"
echo "GPUs allocated on each node: ${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=index,name,uuid --format=csv

# Open MPI obtains the rank count and node allocation directly from Slurm.
time mpirun "${executable}"
