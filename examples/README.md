# Examples

This folder contains examples for the MPI [[1]](#1) and GPU [[2]](#2) implementation of balanced domain decomposition. We also provide a GPU implementation for unbalanced domain decomposition [[3]](#3). Examples using a multiscale GPU Sinkhorn are also provided for reference for both balanced and unbalanced transport. The GPU scripts can be run just by invoking the python interpreter

```bash
python example-domdec-gpu.py
```

For the MPI implementation one needs to invoke the MPI binaries. For example, for running with `openmpi` and 4 workers: 

```bash
mpiexec -n 5 python example-domdec-mpi.py
```
Depending on your configuration, you may need to use the `--oversubscribe` flag.

All the parameters that are set in the scripts can be overriden in the command line. For example, the following runs a larger problem (provided in `examples/data/`) with a larger tolerance:

```bash
python example-domdec-gpu.py --setup_fn1 data/f-000-1024.pickle --setup_fn2 data/f-001-1024.pickle --sinkhorn_error 0.001
```

## References
<a id="1">[1]</a> 
Mauro Bonafini and Bernhard Schmitzer. *Domain decomposition for entropy  regularized optimal transport*. Numerische Mathematik, 149:819–870, 2021.

<a id="2">[2]</a>
Ismael Medina and Bernhard Schmitzer. *Flow updates for decomposition of entropic optimal transport*. arXiv:2405.09400, 2024.

<a id="3">[3]</a>
Ismael Medina, The Sang Nguyen and Bernhard Schmitzer. *Domain decomposition for entropic unbalanced optimal transport*. arXiv:2410.08859, 2024.
