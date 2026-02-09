#!/bin/bash
#SBATCH -J Training_GPU
#SBATCH -o log
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -p batch
#SBATCH --mem=200GB
#SBATCH --gres=gpu:8
#SBATCH -A ptao_ml_protein_dynamics_0001


## personal prefer definition
LOG_FILE=run.log

## using the 'torch' env from anaconda
eval "$(conda shell.bash hook)" 
conda activate geoTDM


## set Wandb to offline mode
## after the job done, can run: wandb sync path/to/your/wandb/run/directory
export WANDB_MODE=offline



# python pt2pdb.py  >> $LOG_FILE
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 16888 \
    experiments/md17_train.py \
    --train_yaml_file configs/md17_qm_train_cond.yaml

# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --master_port 16888 \
#     experiments/md17_train.py \
#     --train_yaml_file configs/md17_train_cond.yaml


# ## sampling conditional md17
# python experiments/md17_sampling.py --eval_yaml_file configs/md17_sampling.yaml

# ## evaluation code
# # python -m experiments.scores --path outputs/md17_aspirin_GeoTDM_uncond_eval/samples.pkl --chem

# echo "run complete on `hostname`: `date` `pwd`" >> ~/logs/finishedJob.log



