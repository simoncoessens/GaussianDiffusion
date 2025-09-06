#!/bin/bash
# Slurm directives to reserve nodes, memory, and time
#SBATCH --job-name=flash_MHA_64h_32_blocks
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=24G
#SBATCH --output=transformer_train_sprites_32x32_resume_sampling_single_image.out

echo "Starting job $SLURM_JOB_ID"

# Load necessary modules (if any)
echo "Loading modules..."
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate gsd
echo "Active conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the Python script
echo "Starting training script..."
cd /gpfs/workdir/coessenss/gsplat
python -m src.train.transformer_train_sprites_32x32_resume_sampling_single_image
