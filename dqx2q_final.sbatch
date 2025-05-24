#!/bin/bash
#SBATCH --job-name=dqx2q_forrest
#SBATCH --partition=dgx2q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load required modules (adjust if you used others)

# Load the environment properly
module purge

# Load only if necessary (PyTorch includes CUDA)
# module load cuda/12.2   # Comment this if it's not found

# Initialize Conda in this non-interactive shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate torchenv

#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH


# Print debug info
echo "SLURM job running on: $(hostname)"
echo "Python path: $(which python)"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print('CUDA:', torch.cuda.is_available())"

# Run the script
cd ~/path/to/your/script
python final_model.py