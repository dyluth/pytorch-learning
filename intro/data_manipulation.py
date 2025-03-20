import os

data_file = os.path.join('data', 'house_tiny.csv')


import pandas as pd

data = pd.read_csv(data_file)
#print(data)

print(data.loc[:,"Price"], "\n\n\n")

inputs = data.iloc[:, :2] # access range by numeric definition - chop off the last column (index 2) - price
print("in:\n",inputs)
print(inputs.shape)

targets= data.loc[:,"Price"]# access column by name
print("tg:\n",targets)
print(targets.shape)
print(targets[0])
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.fillna(inputs.mean())
print(inputs)


import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print(X, "\n",y)