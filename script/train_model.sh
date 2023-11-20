#!/bin/bash
#SBATCH --job-name=Train_GNN
#SBATCH --output=logging/Limconfig-%j.out
#SBATCH --error=logging/Limconfig-%j.error
#SBATCH --chdir /home/suijen/SREX_GNN
#SBATCH --export=ALL
#SBATCH --get-user-env=L

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=08:30:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.j.uijen@tilburguniversity.edu

# Setup modules
if [ ! -z "${SLURM_JOB_ID}" ]; then
    module purge
    module load 2022
    module load Python/3.10.4-GCCcore-11.3.0
    module load aiohttp/3.8.3-GCCcore-11.3.0
    module load matplotlib/3.5.2-foss-2022a
    module load SciPy-bundle/2022.05-foss-2022a
    module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
    module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
fi

# Log versions
echo "PYTHON_VERSION = $(python3.10 --version)"
echo "PIP_VERSION = $(pip3.10 --version)"

# Setup environment
if [ -f .env ]; then
    export $(cat .env | xargs)
fi
export PYTHONPATH=src

# Setup dependencies
pip3.10 install -q \
    tqdm \
    numpy \
    pyvrp \
    torch_geometric \
    torch \


# Run experiment
python3.10 -m main