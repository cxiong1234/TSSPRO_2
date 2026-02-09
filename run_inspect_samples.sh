#!/bin/bash
# Wrapper script to run inspect_samples.py with conda environment activated

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate geoTDM

# Run the inspection script with all arguments passed through
python inspect_samples.py "$@"
