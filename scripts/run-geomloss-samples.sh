#! /usr/bin/bash

# Module load
echo "Loading anaconda"
module load anaconda3
echo "Loading cuda"
module load cuda
echo `nvcc --version`
python run-geomloss-samples.py
