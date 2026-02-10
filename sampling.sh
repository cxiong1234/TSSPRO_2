#!/bin/bash
#SBATCH -J geotdm_train
#SBATCH -o run.log
#SBATCH -e run.log
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH -p batch
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH -A ptao_allostery_0001
#SBATCH --mail-user=alexchen@smu.edu         
#SBATCH --mail-type=begin,end               

module load conda
conda activate geotdm

# ============================================
# GeoTDM Training and Inference
# ============================================

# Choose one of the following modes by uncommenting:

# --- N-Body Conditional Generation (Training + Inference + Eval) ---
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port 16888 \
#     experiments/nbody_train.py \
#     --train_yaml_file configs/nbody_train_cond.yaml

# --- N-Body Unconditional Generation (Training only) ---
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port 16888 \
#     experiments/nbody_train.py \
#     --train_yaml_file configs/nbody_train_uncond.yaml

# --- N-Body Unconditional Sampling (after training) ---
# python experiments/nbody_sampling.py --eval_yaml_file configs/nbody_sampling.yaml

# --- MD17 Conditional Generation ---
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port 16888 \
    experiments/md17_train.py \
    --train_yaml_file configs/md17_train_cond.yaml

# --- ETH Pedestrian Trajectory ---
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port 16888 \
#     experiments/eth_train_new.py \
#     --train_yaml_file configs/eth_train_new.yaml