import re
import matplotlib.pyplot as plt

log_path = '/home/hzheng/jwy/resampling/ablations/cifar100-tiny_images/wrn40-binary-lambda-rand-b_2.0-e_100/console.log'

with open(log_path, 'r') as f:
    logs = f.readlines()

w1s, w2s, w3s = [], [], []
for log_ in logs:
    if log_.startswith('GMM Weights:'):
        w1, w2, w3 = re.findall(r"\d+\.?\d*", log_)
        print(w1, w2, w3)

        w1s.append(float(w1))
        w2s.append(float(w2))
        w3s.append(float(w3))

epochs = range(1, 101)
plt.plot(epochs, w1s, color='k')
plt.plot(epochs, w2s, color='b')
plt.plot(epochs, w3s, color='g')

plt.savefig('random_weights_ab.png')