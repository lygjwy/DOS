# SEA
seeds='3 14 42'

for seed in $seeds; do
    python detect.py --id cifar10 --score abs --pretrain ./esd/s${seed}/cifar10-1_ti_300k/densenet101-t-b_1.0-multistep-r_0.75-k_64/cla_last.pth --gpu_idx 0
done

OODs="svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun"
for seed in $seeds; do
    python detect.py --id cifar100 --ood $OODs --score abs --pretrain ./esd/s${seed}/cifar100-1_ti_300k/densenet101-t-b_1.0-multistep-r_0.75-k_64/cla_last.pth --gpu_idx 0
done