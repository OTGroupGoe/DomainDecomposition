# Examples

This folder contains examples for the MPI and GPU implementation. An example using a multiscale GPU Sinkhorn is also provided for reference. The GPU scripts can be run just by invoking the python interpreter

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
python example-sinkhorn-gpu.py --setup_fn1 data/f-000-1024.pickle --setup_fn2 data/f-001-1024.pickle --sinkhorn_error 0.001
```

