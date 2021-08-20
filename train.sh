#!/bin/sh
#SBATCH -p Aurora
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train.out

python train_38.py
