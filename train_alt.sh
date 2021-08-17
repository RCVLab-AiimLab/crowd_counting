#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train_alt.out

python train_26.py
