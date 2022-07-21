
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
for f in configs/64/*.json;
	do python test_folder/Benchmark2.py $f sys.argv[1] results/dumpBig results/dumpSmall $f;
	echo "run " + f + " done"
done
