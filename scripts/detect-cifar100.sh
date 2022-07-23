# Detect OOD; CIFAR-100 as ID

OODs="svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun"

# Without OE
python detect.py --id cifar100 --oods $OODs --score msp --pretrain ./ckpts/cifar100/wrn40/cla_last.pth --gpu_idx 0

# With OE
## random sample
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100-tiny_images/wrn40-abs-rand/cla_last.pth --gpu_idx 0

## conf sample
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100-tiny_images/wrn40-abs-conf-q_0.0/cla_last.pth --gpu_idx 0
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100-tiny_images/wrn40-abs-conf-q_0.5/cla_last.pth --gpu_idx 0

# dens sample
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100-tiny_images/wrn40-abs-dens-q_0.0/cla_last.pth --gpu_idx 0
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100-tiny_images/wrn40-abs-dens-q_0.5/cla_last.pth --gpu_idx 0