import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

print("random heads/tails from a function:\n")
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])

print("random heads/tails from a torch multinomial:\n")

fair_probs = torch.tensor([0.5, 0.5])
sample = Multinomial(10000, fair_probs).sample()
print(sample) # absolute number of shjeads/tails
print(sample / 10000) # measured probability of heads/tails



counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show() # added by cam to display the graph outside of a jupyter notebook - if running inside of a notebook, need to put %matplotlib inline at the top of the file
