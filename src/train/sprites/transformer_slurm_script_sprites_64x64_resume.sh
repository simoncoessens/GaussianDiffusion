#!/bin/bash
# Slurm directives to reserve nodes, memory, and time
#SBATCH --job-name=sprites_64x64_MHA_64h_12_blocks_16_batches_200ts
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --output=sprites_64x64_MHA_64h_12_blocks_16_batches_200ts_%j.out
#SBATCH --time=23:30:00

echo "Starting job $SLURM_JOB_ID"

# Define paths
WORK_DIR="/gpfs/workdir/coessenss/gsplat/src"
SAVE_PATH="$WORK_DIR/models/Sprites_64x64/SPRITES_64x64_MHA_64h_12blocks_200ts"
mkdir -p "$SAVE_PATH"

# Load necessary modules
echo "Loading modules..."
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate gsd
echo "Active conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Check if we should resume training
RESUME_FLAG=""
if [ -f "$SAVE_PATH/wandb_run_id.txt" ]; then
    echo "Found existing training session, will resume"
    RESUME_FLAG="--resume"
fi

# Run the Python script with resume flag if needed
echo "Starting training script with $RESUME_FLAG"
cd "$WORK_DIR"
python -m train.transformer_train_sprites_64x64_resume $RESUME_FLAG

# Check exit code and submit a new job if training was interrupted
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training job ended before completion (exit code: $EXIT_CODE)"
    
    # Check if we reached the maximum number of epochs
    if [ ! -f "$SAVE_PATH/training_complete.txt" ]; then
        echo "Submitting a new job to continue training"
        cd "$WORK_DIR/train"
        sbatch transformer_slurm_script_sprites_64x64_resume.sh
    fi
fi