#! /usr/bin/bash

# Module load
echo "Loading anaconda"
module load anaconda3/2020.11
echo "Loading cuda"
module load cuda
module load cmake
# This is needed for scipy for some reason
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/sw/rev/21.12/haswell/gcc-9.3.0/anaconda3-2020.11-wtunhc/lib"

# Run job
echo "Running python program"
ipython run-benchmark-softmin.py