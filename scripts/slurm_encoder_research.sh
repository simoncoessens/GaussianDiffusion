#!/bin/bash
#SBATCH --job-name=encoder_research
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/encoder_research_%j.out
#SBATCH --error=logs/encoder_research_%j.err

# Generate all encoder research figures and convergence videos.
# Encodes MNIST with K=70 Gaussians on GPU, runs all experiments, makes videos.
#
# Submit: sbatch scripts/slurm_encoder_research.sh
# Monitor: tail -f logs/encoder_research_<JOBID>.out

echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion

cd "$WORKDIR"
mkdir -p logs reports/encoder_research/figures reports/encoder_research/videos

# Configurable options (override via env vars if needed)
N_PER_CLASS=${N_PER_CLASS:-5}      # images per digit class (50 total)
DEVICE=${DEVICE:-cuda}             # cuda for GPU, cpu as fallback
FRAME_EVERY=${FRAME_EVERY:-10}     # video frame capture interval (epochs)

echo ""
echo "Config:"
echo "  N_PER_CLASS : $N_PER_CLASS  (→ $((N_PER_CLASS * 10)) total images)"
echo "  DEVICE      : $DEVICE"
echo "  FRAME_EVERY : $FRAME_EVERY"
echo "  Output dir  : $WORKDIR/reports/encoder_research/"
echo ""

$PYTHON reports/encoder_research/generate_figures.py \
    --n_per_class  "$N_PER_CLASS" \
    --device       "$DEVICE" \
    --frame_every  "$FRAME_EVERY"

echo ""
echo "Job $SLURM_JOB_ID finished."
echo "Figures: $WORKDIR/reports/encoder_research/figures/"
echo "Videos:  $WORKDIR/reports/encoder_research/videos/"
echo ""
echo "To compile the report (on your local machine):"
echo "  cd reports/encoder_research && pdflatex main.tex && pdflatex main.tex"
