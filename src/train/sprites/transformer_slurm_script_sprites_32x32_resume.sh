#!/bin/bash
# Slurm directives to reserve nodes, memory, and time
#SBATCH --job-name=sprites_MHA_16h_12blocks_32batch_1000epoch
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=32G
#SBATCH --time=24:00:00  # Request full 24 hours
#SBATCH --output=sprites_MHA_16h_16blocks_32batch_1000epoch.out

echo "Starting job $SLURM_JOB_ID at $(date)"

# Load necessary modules
echo "Loading modules..."
module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gsd
echo "Active conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the Python script
echo "Starting training script..."
cd /gpfs/workdir/coessenss/gsplat/src
python -m train.transformer_train_sprites_32x32_resume_sampling

# Check exit code and resubmit if needed
exit_code=$?
if [ $exit_code -eq 0 ]; then
    # Exit code 0 means training was interrupted due to time limit, so resubmit
    echo "Training interrupted due to time limit. Resubmitting job..."
    
    # Get the model directory to check if training is complete
    MODEL_DIR="/gpfs/workdir/coessenss/gsplat/src/models/Sprites_32x32/SPRITES_MHA_16h_12blocks_200ts_mha_classic_1000e"
    CURRENT_EPOCH_FILE="${MODEL_DIR}/current_epoch.txt"
    
    if [ -f "$CURRENT_EPOCH_FILE" ]; then
        current_epoch=$(cat "$CURRENT_EPOCH_FILE")
        echo "Current epoch: $current_epoch"
        
        # Only resubmit if current epoch is less than the target number of epochs
        if [ "$current_epoch" -lt 999 ]; then
            echo "Resubmitting job since training is not complete..."
            sbatch /gpfs/workdir/coessenss/gsplat/src/train/transformer_slurm_script_sprites_32x32_resume_sampling.sh
        else
            echo "Training completed successfully! No need to resubmit."
        fi
    else
        echo "Could not find current epoch file. Resubmitting anyway..."
        sbatch /gpfs/workdir/coessenss/gsplat/src/train/transformer_slurm_script_sprites_32x32_resume_sampling.sh
    fi
else
    echo "Script failed with exit code $exit_code. Not resubmitting."
fi

echo "Job ended at $(date)"