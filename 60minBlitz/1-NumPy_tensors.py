import torch
import numpy as np

data = [[1, 2], [3, 4]]

# create tensor direct from data
x_data = torch.tensor(data)

# create tensor from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# create tensors of the same shape as the original
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# or create a shape, and populate it directly:
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


print(f"\n\nShape of tensor: {rand_tensor.shape}")
print(f"Datatype of tensor: {rand_tensor.dtype}")
print(f"Device tensor is stored on: {rand_tensor.device}")



# We move our tensor to the GPU if available - however we ened to be on the nightly build of torch to make use of this.
# might also need macos metal tools installed to make this work.. this is a sidequest
# if torch.mps.is_available():
#   tensor = rand_tensor.to(torch.device("mps"))
#   print(f"Device tensor is stored on: {tensor.device}")
# else:
#   print("mps not available")

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)