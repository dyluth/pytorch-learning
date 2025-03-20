import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
#print("labels: ", labels)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass

# learning rate of 0.01 and momentum of 0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent

# we will get this warning: [W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
# we can ignore it, as its about optimising on hardware.
# can fix this by moving to use metal and torch.mps to use the M1 GPU

print("done")