#!/bin/bash
#SBATCH --job-name=kofitestjob
#SBATCH --account=def-lantonie 
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=output.txt
module load python/3.10
source myenv/bin/activate 
python remotetest.py


