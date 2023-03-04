import matplotlib.pyplot as plt
import numpy as np

barWidth = 0.3

x = [0.25, 0.5, 0.7, 0.75, 0.8]
bars_sea = [8.71, 7.37, 6.03, 5.75, 5.19]
bars_sea_p = [7.73, 7.08, 5.28, 5.60, 5.95]

r_sea = np.arange(len(bars_sea))
r_sea_p = [x + barWidth for x in r_sea]

plt.clf()

fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)

ax.bar(r_sea, bars_sea, width=barWidth, color='aqua', edgecolor='black', label='SEA')
ax.bar(r_sea_p, bars_sea_p, width=barWidth, color='lime', edgecolor='black', label='SEA+')

ax.tick_params(labelsize=20)
# ax.set_xticks(x, minor=False)
ax.set_xticks([r + 0.5 * barWidth for r in r_sea], ['0.2', '0.5', '0.7', '0.75', '0.8'])

# ax.xaxis.grid(True, which='major', linestyle='dotted')
ax.set_xlabel("OOD Ratio", fontsize=18)

# ax.yaxis.grid(True, which='major', linestyle='dotted')
ax.set_ylabel('FPR95', fontsize=18)

for i in range(len(r_sea)):
    ax.text(x=r_sea[i]-0.1, y=bars_sea[i]+0.1, s=str(bars_sea[i]), size=12)
    ax.text(x=r_sea_p[i]-0.1, y=bars_sea_p[i]+0.1, s=str(bars_sea_p[i]), size=12)

ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.legend(prop={'size': 18})
plt.savefig('cifar10-ood_ratio.png')