import matplotlib.pyplot as plt

ID = 'cifar100'
random_data_path = '/home/hzheng/jwy/resampling/esd/s42/' + ID + '-tiny_images/wrn40-abs-b_1.0-multistep-rand/console.log'
greedy_data_path = '/home/hzheng/jwy/resampling/esd/s42/' + ID + '-tiny_images/wrn40-abs-b_1.0-multistep-greedy_0.0/console.log'
dual_data_path = '/home/hzheng/jwy/resampling/esd/s42/' + ID + '-tiny_images/wrn40-abs-b_1.0-multistep-dual_0.5/console.log'
dsl_data_path = '/home/hzheng/jwy/resampling/esd/s42/' + ID + '-tiny_images/wrn40-abs-b_1.0-multistep-dsl_0.5/console.log'

# read data
def parse_data(data_path):
    proximities, diversities = [], []

    with open(data_path, 'r') as df:
        logs = df.readlines()

    for log in logs:
        if log.startswith('Proximity-Selected:'):
            _, proximity = log.split(' ', 1)
            proximities.append(eval(proximity))
        
        if log.startswith('Diversity:'):
            _, diversity = log.split(' ', 1)
            diversities.append(eval(diversity))
    
    return proximities, diversities

rand_proximities, rand_diversities = parse_data(random_data_path)
greedy_proximities, greedy_diversities = parse_data(greedy_data_path)
dual_proximities, dual_diversities = parse_data(dual_data_path)
dsl_proximities, dsl_diversities = parse_data(dsl_data_path)

x = [10 * i for i in range(1, 11)]

# plot line chart
## proximities
plt.clf()
fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(32, 8), dpi=1000)

ax_p.plot(x, rand_proximities, marker='.', markersize=6, linewidth=4, color='brown', label='Random')
# ax_p.plot(x, greedy_proximities, marker='.', markersize=6, linewidth=2, color='olive', label='Greedy')
ax_p.plot(x, dual_proximities, marker='.', markersize=6, linewidth=4, color='aqua', label='SEA')
ax_p.plot(x, dsl_proximities, marker='.', markersize=6, linewidth=4, color='lime', label='SEA+')
ax_p.tick_params(labelsize=24)
ax_p.set_xticks(x, minor=False)
# ax_p.xaxis.grid(True, which='major', linestyle='dotted')
ax_p.set_xlabel("Epoch", fontsize=24)
# ax_p.yaxis.grid(True, which='major', linestyle='dotted')
ax_p.set_ylabel('Proximity', fontsize=24)
ax_p.legend(prop={'size': 18})
ax_p.spines['top'].set_linewidth(4)
ax_p.spines['bottom'].set_linewidth(4)
ax_p.spines['left'].set_linewidth(4)
ax_p.spines['right'].set_linewidth(4)


ax_d.plot(x, rand_diversities, marker='.', markersize=6, linewidth=4, color='brown', label='Random')
ax_d.plot(x, greedy_diversities, marker='.', markersize=6, linewidth=4, color='olive', label='Greedy')
ax_d.plot(x, dual_diversities, marker='.', markersize=6, linewidth=4, color='aqua', label='SEA')
ax_d.plot(x, dsl_diversities, marker='.', markersize=6, linewidth=4, color='lime', label='SEA+')
ax_d.tick_params(labelsize=24)
ax_d.set_xticks(x, minor=False)
# ax_p.xaxis.grid(True, which='major', linestyle='dotted')
ax_d.set_xlabel("Epoch", fontsize=24)
# ax_p.yaxis.grid(True, which='major', linestyle='dotted')
ax_d.set_ylabel('Diversity', fontsize=24)
ax_d.legend(prop={'size': 18})
ax_d.spines['top'].set_linewidth(4)
ax_d.spines['bottom'].set_linewidth(4)
ax_d.spines['left'].set_linewidth(4)
ax_d.spines['right'].set_linewidth(4)

plt.savefig(ID+'-proximity_diversity.png')