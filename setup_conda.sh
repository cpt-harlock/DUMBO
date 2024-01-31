#!/usr/bin/env bash

# Usage: ./setup_conda.sh

eval "$(conda shell.bash hook)" # Required to use conda in bash

# Create virtual environment with required packages for training the models
rm -r ./conda_envs/training_env # Safety if rerun
conda env create -f ./conda_envs/training_env.yml --prefix ./conda_envs/training_env

# Create virtual environment with required packages for the scheduling simulator
rm -r ./conda_envs/scheduling_env # Safety if rerun
conda env create -f ./conda_envs/scheduling_env.yml --prefix ./conda_envs/scheduling_env

# Install required packages for system Python (i.e., *not* inside a virtual env)
# This is required because the Rust simulator makes Python calls and does not play nice with conda
python -m pip install pandas
python -m pip install numpy
python -m pip install onnxruntime
python -m pip install pickle
python -m pip install json
python -m pip install scikit-learn

