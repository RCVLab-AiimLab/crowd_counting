#!/bin/sh
#SBATCH -p Aurora
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -o train_cont.out

python train_14.py
