#!/bin/bash

CONDA_PREFIX=./venv

# Deactivate and remove existing environment
conda deactivate
conda env remove -p $CONDA_PREFIX

# Create new environment
conda env create --prefix $CONDA_PREFIX --file environment.yml
conda activate $CONDA_PREFIX
