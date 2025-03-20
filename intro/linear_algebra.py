import torch

A = torch.arange(12, dtype=torch.float32).reshape( 3,4)

print(A)
print(A.cumsum(axis=0))
print("\n")
sum_A = A.sum(axis=1, keepdims=True)
print(A, A.shape)
print(sum_A, sum_A.shape)

print(A/sum_A)



x = torch.arange(3, dtype = torch.float32)
y = torch.ones(3, dtype = torch.float32)

print(x,"\n", y,"\n DOT:", torch.dot(x, y))
print(torch.sum(x * y))

# matrix multiplication


A = torch.arange(6, dtype=torch.float32).reshape( 2,3)

print("matrix:\n",A,A.shape, "\n",x, x.shape, torch.mv(A, x), A@x)