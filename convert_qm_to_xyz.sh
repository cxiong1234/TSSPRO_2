#!/bin/bash
# Convert sampled QM region trajectories to XYZ format

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate geoTDM

# Run conversion
# Convert first 10 trajectories as a test (remove --max-traj to convert all)
python convert_to_xyz.py \
    outputs/md17_qm_GeoTDM_cond_eval/samples.pkl \
    --output-dir xyz_qm_region \
    --npz-path data/md17/md17_qm_region.npz \
    --max-traj 10

echo "Conversion complete! XYZ files saved to xyz_qm_region/"
echo "Ground truth: xyz_qm_region/ground_truth/"
echo "Sampled: xyz_qm_region/sampled/"
