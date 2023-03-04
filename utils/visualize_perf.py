import matplotlib.pyplot as plt

metric_str = '> FPR:'

id = 'cifar100'
paradigms = ['rand', 'greedy_0.0', 'dual_0.5', 'dsl_0.5']
metrics_dic ={}

for paradigm in paradigms:
    metrics_dic[paradigm] = []
    data_path = '/home/hzheng/jwy/resampling/' + id + '-' + paradigm + '.txt'

    with open(data_path, 'r') as f:
        logs = f.readlines()

    for log in logs:
        
        if log.startswith(metric_str):
            ood_metrics = log.split(' ')[4:13]
            # print(ood_metrics)
            
            ood_metrics = [float(n) for n in ood_metrics if n]
            metrics_dic[paradigm].append(sum(ood_metrics) / len(ood_metrics))

# draw the plot
x = [5 * i for i in range(1, 21)]
plt.clf()

fig, ax = plt.subplots(figsize=(16, 8), dpi=1000)

ax.plot(x, metrics_dic['rand'], marker='.', markersize=6, linewidth=3, color='brown', label='Random')
ax.plot(x, metrics_dic['greedy_0.0'], marker='.', markersize=6, linewidth=3, color='olive', label='Greedy')
ax.plot(x, metrics_dic['dual_0.5'], marker='.', markersize=6, linewidth=3, color='aqua', label='SEA')
ax.plot(x, metrics_dic['dsl_0.5'], marker='.', markersize=6, linewidth=3, color='lime', label='SEA+')

ax.tick_params(labelsize=20)
ax.set_xticks(x, minor=False)
ax.xaxis.grid(True, which='major', linestyle='dotted')
ax.set_xlabel("Epoch", fontsize=18)

ax.yaxis.grid(True, which='major', linestyle='dotted')
ax.set_ylabel('FPR95', fontsize=18)

ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.legend(prop={'size': 18})
plt.savefig(id+'-fpr.png')