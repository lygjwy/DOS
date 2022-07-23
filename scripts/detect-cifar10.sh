# Detect OOD; CIFAR-10 as ID

# Without OE
python detect.py --id cifar10 --score msp --pretrain ./ckpts/cifar10/wrn40/cla_last.pth --gpu_idx 0

# With OE
## random sample
python detect.py --id cifar10 --pretrain ./ckpts/cifar10-tiny_images/wrn40-abs-rand/cla_last.pth --gpu_idx 0

## conf sample
python detect.py --id cifar10 --pretrain ./ckpts/cifar10-tiny_images/wrn40-abs-conf-q_0.0/cla_last.pth --gpu_idx 0
python detect.py --id cifar10 --pretrain ./ckpts/cifar10-tiny_images/wrn40-abs-conf-q_0.125/cla_last.pth --gpu_idx 0

# dens sample
python detect.py --id cifar10 --pretrain ./ckpts/cifar10-tiny_images/wrn40-abs-dens-q_0.0/cla_last.pth --gpu_idx 0
python detect.py --id cifar10 --pretrain ./ckpts/cifar10-tiny_images/wrn40-abs-dens-q_0.125/cla_last.pth --gpu_idx 0