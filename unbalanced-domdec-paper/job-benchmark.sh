#!/bin/bash
# For running in the SCC cluster
# module load rev/21.12; module load anaconda3 cuda gcc; export PYTHONPATH=""
# For a bit more of memory 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ../examples

# Define integer log2 for layer
log2() {
    local number=$1
    echo "l($number)/l(2)" | bc -l | awk '{print int($1)}'
}

M=0
K=9;

for n in 64 128 256 512 1024;
do
    for i in $( seq $M $K); 
    do
        for j in $( seq $(( i + 1 )) $K)
        do
            for reach in 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8 16
            do                
                files="--setup_fn1 data/f-00$i-$n.pickle --setup_fn2 data/f-00$j-$n.pickle"
                flags="--reach $reach --sinkhorn_error_multiplier 0.25 --max_time 300"
                if [ $n -le 256 ]
                # Last Sinkhorn layer
                then 
                    layer=$(log2 $n)
                    flagsinkhorn="--nLayerSinkhornLast $(( layer - 1 ))"
                else
                    flagsinkhorn="--nLayerSinkhornLast 8"
                fi
                # DomDec 
                python example-domdec-gpu-unbalanced.py $files $flags $flagsinkhorn
                # Sinkhorn
                python example-sinkhorn-gpu-unbalanced.py $files $flags
            done
        done
    done
done