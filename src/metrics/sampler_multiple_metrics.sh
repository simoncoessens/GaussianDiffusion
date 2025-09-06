#!/bin/bash
# Slurm directives to reserve nodes, memory, and time
#SBATCH --job-name=sampler_job
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=32G
#SBATCH --output=sampler_multiple_metrics.out

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

# Check if torchmetrics is installed, if not then install it
# Check if torchmetrics is recognized by the Python interpreter
# echo "Verifying torchmetrics in the Python environment..."
# python -c "import torchmetrics" 2>/dev/null
# pip install torchmetrics[image]
# pip install torch-fidelity
# if [ $? -ne 0 ]; then
#     echo "torchmetrics not recognized, installing via pip..."
#     pip install torchmetrics[image]
#     pip install torch-fidelity
# else
#     echo "torchmetrics package is accessible to Python."
# fi

# Run the Python script
echo "Starting sampling script..."
# python "sampler_multiple_metrics_log_pytorch_fid.py"
# python sampler_multiple_metrics_updated.py
python sampler_multiple_metrics.py