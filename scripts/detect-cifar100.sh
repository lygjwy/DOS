# Detect OOD; CIFAR-100 as ID

OODs="svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun"

# Without OE
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100/wrn40/cla_best.pth --gpu_idx 0

# With OE
## random sample
python detect.py --id cifar100 --oods $OODs --pretrain ./ckpts/cifar100-tiny_images/wrn40-random/cla_last.pth --gpu_idx 0

## weighted resample
for w in conf dens; do
    python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100-tiny_images/wrn40-${w}-woc-wom/cla_last.pth --gpu_idx 0
    python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100-tiny_images/wrn40-${w}-woc-wm/cla_last.pth --gpu_idx 0

    python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100-tiny_images/wrn40-${w}-wc-wom/cla_last.pth --gpu_idx 0
    python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100-tiny_images/wrn40-${w}-wc-wm/cla_last.pth --gpu_idx 0
done