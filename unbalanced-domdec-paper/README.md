# Domain decomposition for entropic unbalanced optimal transport: numerical experiments

This folder contains the numerical experiments appearing in the arXiv preprint [[1]](#1), for reproducibility purposes. For the installation of the required libraries we refer to the `README.md` in the parent directory. The files are structured as follows: 


* `PD-gap-evolution-unbalanced-sinkhorn.ipynb` contains the code needed to reproduce Figure 1.
* `compare-parallelization-strategies.ipynb` contains the code needed to reproduce Figures 2 and 3.
* `deformation-map.ipynb` contains the code needed to reproduce Figure 6. By default it reads some precomputed datafiles. One may recompute them by running `job-deformation-map.sh`.
* `benchmark.ipynb` contains the code needed to reproduce Figure 7 and Table 2. By default it reads the result files generated during the work on [[1]](#1). One may reproduce these results by running `job-benchmark.sh`. In our case it took about 24 hours on a V100 GPU.


<a id="1">[1]</a>
Ismael Medina, The Sang Nguyen and Bernhard Schmitzer. *Domain decomposition for entropic unbalanced optimal transport*. arXiv:2410.08859, 2024.