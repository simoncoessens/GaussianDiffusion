#!/bin/bash
# Quick status check for all running experiments
PROJECT=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$PROJECT"

echo "=== Status at $(date +%H:%M) ==="
echo ""

# SLURM jobs
echo "--- Running jobs ---"
squeue -u coessenss --format="%.12i %.20j %.10T %.10M" 2>/dev/null
echo ""

# Encoding progress
echo "--- Encoding shards ---"
for i in 0 1 2 3; do
  log="logs/encode_full_132390_${i}.out"
  if [ -f "$log" ]; then
    last=$(tail -1 "$log" 2>/dev/null)
    echo "  Shard $i: $last"
  fi
done
echo ""

# Full dataset
if [ -f "data/mnist/mnist_gaussians_K70.h5" ]; then
  echo "*** FULL DATASET READY: data/mnist/mnist_gaussians_K70.h5 ***"
  ls -lh "data/mnist/mnist_gaussians_K70.h5"
else
  echo "Full dataset: NOT YET READY"
fi
echo ""

# Small model progress
echo "--- Small model training ---"
for job in 133139 133219 133220; do
  log="logs/small_model_${job}.out"
  if [ -f "$log" ]; then
    config=$(head -10 "$log" | grep -oP "\d+B/\d+H/\d+D.*")
    epoch=$(grep -oP "Epoch \d+" "$log" | tail -1)
    echo "  Job $job ($config): $epoch"
  fi
done
echo ""

# All completed metrics (sorted by FID)
echo "--- All FID metrics (sorted) ---"
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
for d in samples/mnist/*/metrics.json; do
  tag=$(echo "$d" | sed 's|samples/mnist/||;s|/metrics.json||')
  fid=$($PYTHON -c "import json; d=json.load(open('$d')); k='FID' if 'FID' in d else 'fid'; print(f'{d[k]:.2f}')" 2>/dev/null)
  if [ -n "$fid" ]; then
    echo "  $fid  $tag"
  fi
done | sort -n
