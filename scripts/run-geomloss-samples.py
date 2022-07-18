import torch
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import pykeops

# Clear ~/.cache/pykeops2.1/...
print("Cleaning previous config")
pykeops.clean_pykeops()
# Rebuild from scratch the required binaries
print("Rebuild binaries")
pykeops.test_torch_bindings()

# Create some large point clouds in 3D
x = torch.randn(100000, 3).cuda()
y = torch.randn(200000, 3).cuda()

# Define a Sinkhorn (~Wasserstein) loss between sampled measures
loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
print(loss)
L = loss(x, y)  # By default, use constant weights = 1/number of samples
print(L)
