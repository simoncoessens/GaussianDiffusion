#!/bin/bash
#SBATCH --job-name=merge_mnist
#SBATCH --mail-type=END,FAIL
#SBATCH -p cpu_med
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_mnist_%j.out
#SBATCH --error=logs/merge_mnist_%j.err

# Merge 4 Gaussian shard HDF5 files into one master file.
# Should be submitted with a dependency on all encode jobs finishing:
#
#   JID=$(sbatch --array=0-3 scripts/mnist/slurm/encode.sh | awk '{print $NF}')
#   sbatch --dependency=afterok:${JID} scripts/mnist/slurm/merge.sh
#
# Or, submit manually after all shards complete:
#   sbatch scripts/mnist/slurm/merge.sh

echo "========================================================"
echo "  merge_mnist  job ${SLURM_JOB_ID}"
echo "  Host    : $(hostname)"
echo "  Started : $(date)"
echo "========================================================"

module load gcc/15.1.0/gcc-15.1.0
module load anaconda3/2023.09-0/none-none

export PYTHONUNBUFFERED=1

# Use the NLS env (has h5py installed)
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python

if [ ! -f "$PYTHON" ]; then
    echo "[FATAL] Python not found: $PYTHON"
    exit 1
fi

echo "  Python  : $PYTHON"

WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs data

$PYTHON scripts/mnist/merge_shards.py \
    --shard_dir data/mnist/shards \
    --out_file  data/mnist/mnist_gaussians_K70.h5 \
    --n_shards  4 \
    --n_total   70000

EXIT_CODE=$?
echo ""
echo "========================================================"
echo "  Merge finished with exit code $EXIT_CODE"
echo "  Finished: $(date)"
echo "========================================================"
exit $EXIT_CODE
