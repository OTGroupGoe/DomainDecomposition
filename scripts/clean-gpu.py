import pykeops

# Clear ~/.cache/pykeops2.1/...
print("Cleaning previous config")
pykeops.clean_pykeops()
# Rebuild from scratch the required binaries
print("Rebuild binaries")
pykeops.test_torch_bindings()
