#!/bin/sh
#SBATCH -p Aurora
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train_alt.out

python train_37.py
