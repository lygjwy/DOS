OODs="svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun"
# CIFAR-100
# python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100/wrn40/cla_best.pth --gpu_idx 0

# random sample
# for p in tune train; do
#     for t in fix_loc var_loc fix_rand var_rand; do
#         python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100-tiny_images/wrn40-${p}-${t}/cla_last.pth --gpu_idx 0
#     done
# done

# weighted resample
for p in tune train; do
    for w in prob logit; do
        for t in fix var; do
            for r in wr wor; do
                python detect.py --id cifar100 --oods $OODs --pretrain ./outputs/cifar100-tiny_images/wrn40-${p}-weighted-${w}-${t}-${r}/cla_last.pth --gpu_idx 0
            done
        done
    done
done