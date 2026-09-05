#!/bin/bash -x
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --job-name=lbm
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Use the same modules here that were loaded when configuring and compiling.
module purge
module load toolkit/nvidia-hpc-sdk/25.1
module load mpi/openmpi/5.0-nvidia-25.1-nompi

# Work around UCX misclassifying CUDA allocations as host memory. Without
# this, intra-node CUDA-aware MPI transfers can segfault in uct_mm_ep_am_bcopy.
export UCX_MEMTYPE_CACHE=n

executable="${SLURM_SUBMIT_DIR}/build-a100/executables/main"
run_directory="${SLURM_SUBMIT_DIR}/results/${SLURM_JOB_ID}"

if [[ ! -x "${executable}" ]]; then
    echo "Executable not found: ${executable}" >&2
    echo "Build it first with: cmake --preset a100-release && cmake --build --preset a100-release -j" >&2
    exit 1
fi

# The program writes to relative paths under out/data. Running in a directory
# unique to this Slurm job prevents a later run from overwriting those CSVs.
mkdir -p "${run_directory}"
cd "${run_directory}"

echo "Running ${SLURM_NTASKS} MPI rank(s) on ${SLURM_JOB_NUM_NODES} node(s)."
echo "Results will be stored in ${run_directory}/out/data."
nvidia-smi --query-gpu=index,name,uuid --format=csv

time mpirun --bind-to none "${executable}"
