# Domain decomposition for entropic optimal transport

This folder contains the implementation of the domain decomposition algorithm for optimal transport. We cover the original implementation with CPU parallelization in MPI [[1]](#1), the later GPU implementation [[2]](#2) and finally the adaptation to unbalanced transport, also based on GPUs [[3]](#3). 

The principal routines are found on the following files: 

* `DomainDecomposition.py`: Defines the basis for (sequential) domain decomposition on CPUs.
* `DomDecParallel.py` and `DomDecParallelMPI.py`: Parallel MPI version.
* `DomainDecompositionGPU.py`: GPU implementation for balanced transport. 
* `DomDecUnbalancedGPU.py`: GPU implementation for unbalanced transport.

## References
<a id="1">[1]</a> 
Mauro Bonafini and Bernhard Schmitzer. *Domain decomposition for entropy  regularized optimal transport*. Numerische Mathematik, 149:819–870, 2021.

<a id="2">[2]</a>
Ismael Medina and Bernhard Schmitzer. *Flow updates for decomposition of entropic optimal transport*. arXiv:2405.09400, 2024.

<a id="3">[3]</a>
Ismael Medina, The Sang Nguyen and Bernhard Schmitzer. *Domain decomposition for entropic unbalanced optimal transport*. arXiv:2410.08859, 2024.
