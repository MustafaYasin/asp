#!/bin/bash

#SBATCH --job-name=run
#SBATCH --partition=All
#SBATCH --mail-type=all
#SBATCH --mail-user=mao.yang@campus.lmu.de
#SBATCH --output=slurm-%A.out

date
hostname

source /home/m/mao/.bashrc
conda activate tennis-old
python /home/m/mao/workspace/asp/train.py
