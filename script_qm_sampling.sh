#!/bin/bash
#SBATCH -J QM_Sampling_GPU
#SBATCH -o sampling_log
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p batch
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH -A ptao_ml_protein_dynamics_0001


## personal prefer definition
LOG_FILE=sampling_run.log

## using the 'torch' env from anaconda
eval "$(conda shell.bash hook)" 
conda activate geoTDM


## set Wandb to offline mode
## after the job done, can run: wandb sync path/to/your/wandb/run/directory
export WANDB_MODE=offline


## QM region sampling
python experiments/md17_sampling.py --eval_yaml_file configs/md17_qm_sampling.yaml --device 0

echo "Sampling complete on `hostname`: `date` `pwd`" >> ~/logs/finishedJob.log


