import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.pyplot as plt
import random

m1 = (-4, -3)
c1 = [[3, 2], [2, 3]]

m2 = (-1, 3)
c2 = [[2, 1], [1, 4]]

m3 = (3, -1)
c3 = [[4, -2], [-2, 2]]

id_rv_1 = multivariate_normal(m1, c1)
id_rv_2 = multivariate_normal(m2, c2)
id_rv_3 = multivariate_normal(m3, c3)

id_d1 = id_rv_1.rvs(size=1000)
id_x1, id_y1 = id_d1[:, 0], id_d1[:, 1]
id_d2 = id_rv_2.rvs(size=1000)
id_x2, id_y2 = id_d2[:, 0], id_d2[:, 1]
id_d3 = id_rv_3.rvs(size=1000)
id_x3, id_y3 = id_d3[:, 0], id_d3[:, 1]

id_x = np.concatenate((id_x1, id_x2, id_x3), axis=0)
id_y = np.concatenate((id_y1, id_y2, id_y3), axis=0)
id_label = ['id'] * len(id_x)

# OOD data
ood_ms = (np.random.rand(300, 2) - 0.5) * 2 * 10
ood_cs = []
for i in range(300):
    A = np.random.rand(2, 2)
    ood_cs.append(np.dot(A, A.T))

selected_idxs = []
ood_x, ood_y = [], []
# OOD samples
idx = 0
for ood_m, ood_c in zip(ood_ms, ood_cs):
    
    ood_rv = multivariate_normal(ood_m, ood_c)
    if (id_rv_1.pdf(ood_m) + id_rv_2.pdf(ood_m) + id_rv_3.pdf(ood_m)) < 0.01:
        ood_d = ood_rv.rvs(size=20)
        ood_x.append(ood_d[:, 0])
        ood_y.append(ood_d[:, 1])
        selected_idxs.append(idx)
    idx += 1
ood_ms = ood_ms[selected_idxs]
ood_cs = np.array(ood_cs)[selected_idxs]

ood_x = np.concatenate(ood_x, axis=0)
ood_y = np.concatenate(ood_y, axis=0)
ood_label = ['ood'] * len(ood_x)

data = {'x': np.concatenate([id_x, ood_x], axis=0), 'y': np.concatenate([id_y, ood_y], axis=0), 'label': np.concatenate([id_label, ood_label], axis=0)}
# sns.kdeplot(x=id_x, y=id_y, cmap='BuGn', fill=True, levels=10, bw_method=0.3, bw_adjust=1.0, alpha=1.0)

# all scatter plot
sid_idx = np.asarray(random.sample(list(range(len(id_x))), k=100), dtype=int)
sid_x = id_x[sid_idx]
sid_y = id_y[sid_idx]
sid_label = np.asarray(id_label)[sid_idx]

sood_idx = np.asarray(random.sample(list(range(len(ood_x))), k=200), dtype=int)
sood_x = ood_x[sood_idx]
sood_y = ood_y[sood_idx]
sood_label = np.asarray(ood_label)[sood_idx]

data = {'x': np.concatenate([sid_x, sood_x], axis=0), 'y': np.concatenate([sid_y, sood_y], axis=0), 'label': np.concatenate([sid_label, sood_label], axis=0)}
sns.scatterplot(data=data, x='x', y='y', hue='label', hue_order=['id', 'ood'], palette=['green', 'red'], style='label', style_order=['id', 'ood'], markers=['+', 'x'], legend=False)
plt.axis('off')
plt.savefig('contour.png', dpi=1000)

# random
plt.clf()
sns.kdeplot(x=id_x, y=id_y, cmap='BuGn', fill=True, levels=10, bw_method=0.3, bw_adjust=1.0, alpha=1.0)

r_sood_idx = np.asarray(random.sample(list(range(len(sood_x))), k=30), dtype=int)
ru_sood_idx = [idx for idx in range(200) if idx not in r_sood_idx]

r_sood_x = sood_x[r_sood_idx]
r_sood_y = sood_y[r_sood_idx]

ru_sood_x = sood_x[ru_sood_idx]
ru_sood_y = sood_y[ru_sood_idx]

rsood_x = np.concatenate([r_sood_x, ru_sood_x], axis=0)
rsood_y = np.concatenate([r_sood_y, ru_sood_y], axis=0)
rsood_sampled = (['ood_sampled'] * len(r_sood_idx))
rsood_sampled.extend(['ood_unsampled'] * len(ru_sood_idx))

sid_sampled = (['id_sampled'] * len(sid_idx))

data = {'x': np.concatenate([sid_x, rsood_x], axis=0), 'y': np.concatenate([sid_y, rsood_y], axis=0), 'label': np.concatenate([sid_label, sood_label], axis=0), 'sampled': np.concatenate([sid_sampled, rsood_sampled], axis=0)}
sns.scatterplot(data=data, x='x', y='y', hue='sampled', hue_order=['id_sampled', 'ood_sampled', 'ood_unsampled'], palette=['green', 'red', 'grey'], style='label', style_order=['id', 'ood'], markers=['+', 'x'], legend=False)

plt.axis('off')
# plt.savefig('contour-random.png', dpi=1000)

# greedy
plt.clf()
sns.kdeplot(x=id_x, y=id_y, cmap='BuGn', fill=True, levels=10, bw_method=0.3, bw_adjust=1.0, alpha=1.0)

id_score = []
for i in range(200):
    sood_i = [sood_x[i], sood_y[i]]
    id_score.append(id_rv_1.pdf(sood_i) + id_rv_2.pdf(sood_i) + id_rv_3.pdf(sood_i))
gu_sood_idx = np.argsort(id_score)[:170]
g_sood_idx = np.argsort(id_score)[170:]

g_sood_x = sood_x[g_sood_idx]
g_sood_y = sood_y[g_sood_idx]

gu_sood_x = sood_x[gu_sood_idx]
gu_sood_y = sood_y[gu_sood_idx]

gsood_x = np.concatenate([g_sood_x, gu_sood_x], axis=0)
gsood_y = np.concatenate([g_sood_y, gu_sood_y], axis=0)
gsood_sampled = (['ood_sampled'] * len(g_sood_idx))
gsood_sampled.extend(['ood_unsampled'] * len(gu_sood_idx))

# sid_sampled = (['id_sampled'] * len(sid_idx))

data = {'x': np.concatenate([sid_x, gsood_x], axis=0), 'y': np.concatenate([sid_y, gsood_y], axis=0), 'label': np.concatenate([sid_label, sood_label], axis=0), 'sampled': np.concatenate([sid_sampled, gsood_sampled], axis=0)}
sns.scatterplot(data=data, x='x', y='y', hue='sampled', hue_order=['id_sampled', 'ood_sampled', 'ood_unsampled'], palette=['green', 'red', 'grey'], style='label', style_order=['id', 'ood'], markers=['+', 'x'], legend=False)

plt.axis('off')
# plt.savefig('contour-greedy.png', dpi=1000)

# dual
plt.clf()
sns.kdeplot(x=id_x, y=id_y, cmap='BuGn', fill=True, levels=10, bw_method=0.3, bw_adjust=1.0, alpha=1.0)

id_score = []
for i in range(200):
    sood_i = [sood_x[i], sood_y[i]]
    id_score.append(id_rv_1.pdf(sood_i) + id_rv_2.pdf(sood_i) + id_rv_3.pdf(sood_i))
d_sood_idx = np.asarray(random.sample(list(np.argsort(id_score)[140:]), k=30), dtype=int)
du_sood_idx = [idx for idx in range(200) if idx not in d_sood_idx]

d_sood_x = sood_x[d_sood_idx]
d_sood_y = sood_y[d_sood_idx]

du_sood_x = sood_x[du_sood_idx]
du_sood_y = sood_y[du_sood_idx]

dsood_x = np.concatenate([d_sood_x, du_sood_x], axis=0)
dsood_y = np.concatenate([d_sood_y, du_sood_y], axis=0)
dsood_sampled = (['ood_sampled'] * len(d_sood_idx))
dsood_sampled.extend(['ood_unsampled'] * len(du_sood_idx))

data = {'x': np.concatenate([sid_x, dsood_x], axis=0), 'y': np.concatenate([sid_y, dsood_y], axis=0), 'label': np.concatenate([sid_label, sood_label], axis=0), 'sampled': np.concatenate([sid_sampled, dsood_sampled], axis=0)}
sns.scatterplot(data=data, x='x', y='y', hue='sampled', hue_order=['id_sampled', 'ood_sampled', 'ood_unsampled'], palette=['green', 'red', 'grey'], style='label', style_order=['id', 'ood'], markers=['+', 'x'], legend=False)

plt.axis('off')
# plt.savefig('contour-dual.png', dpi=1000)