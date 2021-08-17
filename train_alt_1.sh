#!/bin/sh
#SBATCH -p Aurora
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train_alt_1.out

python train_24.py
