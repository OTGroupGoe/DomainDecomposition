#!/bin/bash
# For running in the SCC cluster
# module load rev/21.12; module load anaconda3 cuda gcc; export PYTHONPATH=""
# For a bit more of memory 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ../examples

for reach in 0.015625 0.0625 0.25 1 4 16
do                
    files="--setup_fn1 data/f-000-256.pickle --setup_fn2 data/f-001-256.pickle"
    flags="--reach $reach --sinkhorn_error_multiplier 0.25 --max_time 300 --nLayerSinkhornLast 7"
    dump="--aux_dump_finest True"
    # DomDec 
    python example-domdec-gpu-unbalanced.py $files $flags $flagsinkhorn $dump
done
