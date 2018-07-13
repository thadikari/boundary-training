#!/bin/bash
#SBATCH --account=def-sdraper
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=07:00:00

pwd
cd /scratch/s/sdraper/tharindu/projects/boundary_training
pwd
mkdir -p runs
cd runs
pwd

source ../src/niagara/setup_env.sh
echo 'executing run'
srun python -u ../src/arrmpijob.py $SLURM_ARRAY_TASK_ID
echo 'done run!'
