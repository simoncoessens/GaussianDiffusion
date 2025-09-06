#!/bin/bash
# Slurm directives to reserve nodes, memory, and time
#SBATCH --job-name=sampler_job
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpu_test
#SBATCH --mem=32G
#SBATCH --output=sampler_sprites.out

echo "Starting job $SLURM_JOB_ID"

# Load necessary modules (if any)
echo "Loading modules..."
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Activate your conda environment
source activate gsd
echo "Conda environment activated"

# Check the active conda environment
echo "Active conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the Python script
echo "Starting sampling script..."
# python "sampler_sprites.py"
cd /gpfs/workdir/coessenss/gsplat
python -m src.metrics.sampler_sprites
#python sampler_sprites_multiple.py
