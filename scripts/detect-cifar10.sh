# Detect OOD; CIFAR-10 as ID

# Without OE
# python detect.py --id cifar10 --pretrain ./outputs/cifar10/wrn40/cla_best.pth --gpu_idx 0

# With OE
## random sample
python detect.py --id cifar10 --pretrain ./outputs/cifar10-tiny_images/wrn40-random/cla_last.pth --gpu_idx 0

## weighted sample
for w in conf dens; do
    python detect.py --id cifar10 --pretrain ./outputs/cifar10-tiny_images/wrn40-${w}-woc-wom/cla_last.pth --gpu_idx 0
    python detect.py --id cifar10 --pretrain ./outputs/cifar10-tiny_images/wrn40-${w}-woc-wm/cla_last.pth --gpu_idx 0

    python detect.py --id cifar10 --pretrain ./outputs/cifar10-tiny_images/wrn40-${w}-wc-wom/cla_last.pth --gpu_idx 0
    python detect.py --id cifar10 --pretrain ./outputs/cifar10-tiny_images/wrn40-${w}-wc-wm/cla_last.pth --gpu_idx 0
done