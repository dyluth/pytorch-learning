import torch

x = torch.arange(4.0)
print(x)

# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
print(x.grad)  # The gradient is None by default

print("2*x*x: ", 2*x*x)
print(torch.dot(x, x))
y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

print(x.grad == 4 * x)



x.grad.zero_()  # Reset the gradient
y = x.sum()
print("\n\n", x,"\n", y)
print(y.backward())
print(x.grad)

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
print("\n\ny: ",y,"\nbackwards: ", x.grad)


x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

print("\n\ndetatch u, so the provenance from y is forgotten - for optimisation reasons\n")
z.sum().backward()
print("\nz:",z,"\nu:",u, "\nx:", x,"\n",x.grad == u)

