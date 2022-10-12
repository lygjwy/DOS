import numpy as np

seed = 2
epochs = 100

epoch_seeds = [seed]
for i in range(epochs-1):
    np.random.seed(epoch_seeds[-1])
    epoch_seeds.append(np.random.randint(1000 * i, 1000 * (i+1)))

print(epoch_seeds)