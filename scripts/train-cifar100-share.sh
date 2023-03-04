# Train CIFAR100 CLF with OE

factors='1'
seeds='3 14 42'

for factor in $factors; do
    for seed in $seeds; do
        python train_rand_epoch.py --size_factor_sampled_ood $factor --seed $seed --training abs --beta 1.0 --id cifar100 --ood ti_300k --gpu_idx 1
    done
done