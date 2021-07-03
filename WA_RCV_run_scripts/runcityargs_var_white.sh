#!/bin/sh
#SBATCH -J cityarg
#SBATCH -p batch #run on batch queue
#SBATCH--time=2-23:00:00 #day-hour:minute:second
#SBATCH -n 1 #request cores
#SBATCH --mem=10000 #request memory
python -u run_all_models_with_sys_args_var_white.py $1 > $2