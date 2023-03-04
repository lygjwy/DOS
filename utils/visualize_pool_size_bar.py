import matplotlib.pyplot as plt
import numpy as np

barWidth = 0.3

x = [262144, 524288, 1048574]

bars_sea_fpr95 = [46.39, 45.67, 45.85]
bars_sea_p_fpr95 = [45.28, 45.67, 41.99]

bars_sea_auroc = [91.05, 91.38, 91.26]
bars_sea_p_auroc = [91.24, 91.17, 91.97]

bars_sea_aupr = [92.10, 92.64, 92.43]
bars_sea_p_aupr = [92.48, 92.24, 93.04]

r_sea = np.arange(len(bars_sea_fpr95))
r_sea_p = [x + barWidth for x in r_sea]

plt.clf()

fig, (ax_fpr95, ax_auroc, ax_aupr) = plt.subplots(1, 3, figsize=(27, 6), dpi=1000)

ax_fpr95.bar(r_sea, bars_sea_fpr95, width=barWidth, color='aqua', edgecolor='black', label='SEA')
ax_fpr95.bar(r_sea_p, bars_sea_p_fpr95, width=barWidth, color='lime', edgecolor='black', label='SEA+')

ax_auroc.bar(r_sea, bars_sea_auroc, width=barWidth, color='aqua', edgecolor='black', label='SEA')
ax_auroc.bar(r_sea_p, bars_sea_p_auroc, width=barWidth, color='lime', edgecolor='black', label='SEA+')

ax_aupr.bar(r_sea, bars_sea_aupr, width=barWidth, color='aqua', edgecolor='black', label='SEA')
ax_aupr.bar(r_sea_p, bars_sea_p_aupr, width=barWidth, color='lime', edgecolor='black', label='SEA+')

ax_fpr95.tick_params(labelsize=20)
ax_fpr95.set_xticks([r + 0.5 * barWidth for r in r_sea], ['262144', '524288', '1048574'])
ax_fpr95.set_xlabel("Size of Candidate Anomaly Data", fontsize=18)
ax_fpr95.set_ylabel('FPR95', fontsize=18)

ax_auroc.tick_params(labelsize=20)
ax_auroc.set_xticks([r + 0.5 * barWidth for r in r_sea], ['262144', '524288', '1048574'])
ax_auroc.set_xlabel("Size of Candidate Anomaly Data", fontsize=18)
ax_auroc.set_ylabel('AUROC', fontsize=18)

ax_aupr.tick_params(labelsize=20)
ax_aupr.set_xticks([r + 0.5 * barWidth for r in r_sea], ['262144', '524288', '1048574'])
ax_aupr.set_xlabel("Size of Candidate Anomaly Data", fontsize=18)
ax_aupr.set_ylabel('AUPR', fontsize=18)

for i in range(len(r_sea)):
    ax_fpr95.text(x=r_sea[i]-0.1, y=bars_sea_fpr95[i]+0.1, s=str(bars_sea_fpr95[i]), size=12)
    ax_fpr95.text(x=r_sea_p[i]-0.1, y=bars_sea_p_fpr95[i]+0.1, s=str(bars_sea_p_fpr95[i]), size=12)

    ax_auroc.text(x=r_sea[i]-0.1, y=bars_sea_auroc[i]+0.1, s=str(bars_sea_auroc[i]), size=12)
    ax_auroc.text(x=r_sea_p[i]-0.1, y=bars_sea_p_auroc[i]+0.1, s=str(bars_sea_p_auroc[i]), size=12)

    ax_aupr.text(x=r_sea[i]-0.1, y=bars_sea_aupr[i]+0.1, s=str(bars_sea_aupr[i]), size=12)
    ax_aupr.text(x=r_sea_p[i]-0.1, y=bars_sea_p_aupr[i]+0.1, s=str(bars_sea_p_aupr[i]), size=12)

ax_fpr95.spines['top'].set_linewidth(2)
ax_fpr95.spines['bottom'].set_linewidth(2)
ax_fpr95.spines['left'].set_linewidth(2)
ax_fpr95.spines['right'].set_linewidth(2)

ax_auroc.spines['top'].set_linewidth(2)
ax_auroc.spines['bottom'].set_linewidth(2)
ax_auroc.spines['left'].set_linewidth(2)
ax_auroc.spines['right'].set_linewidth(2)

ax_aupr.spines['top'].set_linewidth(2)
ax_aupr.spines['bottom'].set_linewidth(2)
ax_aupr.spines['left'].set_linewidth(2)
ax_aupr.spines['right'].set_linewidth(2)

box = ax_auroc.get_position()
# ax_fpr95.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put a legend below current axis
ax_auroc.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, prop={'size': 18})

# plt.legend(prop={'size': 18}, bbox_to_anchor=(1.0, 1.0))
plt.savefig('cifar100-pool_size.png')