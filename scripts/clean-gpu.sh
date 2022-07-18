#! /usr/bin/bash

# Module load
echo "Loading anaconda"
module load anaconda3/2020.11
echo "Loading cuda"
module load cuda
python clean-gpu.py
