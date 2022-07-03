# CIFAR-100
python detect.py --id cifar100 --oods svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun --pretrain ./outputs/cifar100/wrn40/cla_best.pth --gpu_idx 0
for p in tune train; do
    for t in fix_loc fix_rand var_loc var_rand; do
        python detect.py --id cifar100 --oods svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun --pretrain ./outputs/cifar100-tiny_images/wrn40-${p}-${t}/cla_last.pth --gpu_idx 0
    done
done

