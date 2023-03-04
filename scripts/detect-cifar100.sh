# Detect OOD; CIFAR-100 as ID

OODs="svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun"
seeds='3 14 42'

# OE
for seed in $seeds; do
    python detect.py --id cifar100 --oods $OODs --score msp --pretrain ./esd/s${seed}/cifar100-1_ti_300k/densenet101-uni-b_0.5-multistep-rand_epoch/cla_last.pth --gpu_idx 1
done

# Energy
for seed in $seeds; do
    python detect.py --id cifar100 --oods $OODs --score energy --pretrain ./esd/s${seed}/cifar100-1_ti_300k/densenet101-energy-b_0.1-multistep-rand_epoch/cla_last.pth --gpu_idx 1
done

# NTOM
for seed in $seeds; do
    python detect.py --id cifar100 --oods $OODs --score abs --pretrain ./esd/s${seed}/cifar100-1_ti_300k/densenet101-abs-b_1.0-multistep-greedy_0.0/cla_last.pth --gpu_idx 1
done

# Share
for seed in $seeds; do
    python detect.py --id cifar100 --oods $OODs --score abs --pretrain ./esd/s${seed}/cifar100-1_ti_300k/densenet101-abs-b_1.0-multistep-rand_epoch/cla_last.pth --gpu_idx 1
done

# POEM
for seed in $seeds; do
    python detect.py --id cifar100 --oods $OODs --score energy --pretrain ../poem/esd/s${seed}/cifar100-1_ti_300k/densenet101-b_0.1/100.pth --gpu_idx 1
done