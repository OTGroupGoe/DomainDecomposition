# Domain decomposition for entropic unbalanced optimal transport: numerical experiments

This folder contains the numerical experiments appearing in the arXiv preprint [[1]](#1), for reproducibility purposes. For the installation of the required libraries we refer to the `README.md` in the parent directory. The files are structured as follows: 


* `PD-gap-evolution-unbalanced-sinkhorn.ipynb` contains the code needed to reproduce Figure 1.
* `compare-parallelization-strategies.ipynb` contains the code needed to reproduce Figures 2 and 3.
* `deformation-map.ipynb` contains the code needed to reproduce Figure 6. By default it reads some precomputed datafiles. One may opt to recompute them by running `job-deformation-map.sh`. This will run the file `../examples/example-domdec-gpu-unbalanced.py`, for a reference problem with resolution $N = 256$ and several values of the soft-marginal penalty $\lambda$.

* `benchmark.ipynb` contains the code needed to reproduce Figure 7 and Table 2. By default it reads the result files generated during the work on [[1]](#1). One may recompute these results by running `job-benchmark.sh`.  This will call the files `../examples/example-domdec-gpu-unbalanced.py` and `../examples/example-sinkhorn-gpu-unbalanced.py`, for problem sizes ranging from $64\times 64$ to $1024\times 1024$, and values $\sqrt{\lambda}$ from $1/64$ to $16$. In our case running the whole benchmark suite took about 24 hours on a V100 GPU.


<a id="1">[1]</a>
Ismael Medina, The Sang Nguyen and Bernhard Schmitzer. *Domain decomposition for entropic unbalanced optimal transport*. arXiv:2410.08859, 2024.