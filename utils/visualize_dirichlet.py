import numpy as np
import matplotlib.pyplot as plt

w1s, w2s, w3s = np.random.dirichlet((20, 55, 25), 100).transpose()

epochs = range(1, 101)
plt.plot(epochs, w1s, color='k')
plt.plot(epochs, w2s, color='b')
plt.plot(epochs, w3s, color='g')

plt.savefig('dirichlet_weights.png')
