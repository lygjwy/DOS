# CIFAR-10
python detect.py --id cifar10 --pretrain ./outputs/cifar10/wrn40/cla_best.pth --gpu_idx 0
for p in tune train; do
    for t in fix_loc fix_rand var_loc var_rand; do
        python detect.py --id cifar10 --pretrain ./outputs/cifar10-tiny_images/wrn40-${p}-${t}/cla_last.pth --gpu_idx 0
    done
done