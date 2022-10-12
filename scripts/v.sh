OODs="svhn lsunc dtd places365_10k cifar10 tinc lsunr tinr isun"

for N in $(seq 80 89); do
    python detect.py --id cifar100 --oods $OODs --score binary --include_binary --pretrain ./outputs/cifar100-tiny_images/wrn40-binary-lambda-gmm-b_2.0-wp_0-c1e_1.0-c2e_min_0.0-c2e_max_0.0-c3e_0.0/${N}.pth --fig_name w1-${N}.png
    # python detect.py --id cifar100 --oods $OODs --score binary --include_binary --pretrain ./snapshots/cifar100-tiny_images/wrn40-binary-lambda-rand-b_2.0-e_100/${N}.pth --fig_name demo-${N}.png
done