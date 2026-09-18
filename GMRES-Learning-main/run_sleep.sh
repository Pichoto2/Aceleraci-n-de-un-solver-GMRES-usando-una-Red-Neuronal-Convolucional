#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
####SBATCH --cpus-per-task=5
#SBATCH --partition=volta
#SBATCH --nodelist=acp03
source ~/.bashrc
conda activate myenv
python sleep.py
