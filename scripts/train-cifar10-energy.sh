# Train CIFAR10 CLF with Energy

factors='1'
seeds='3 14 42'

for factor in $factors; do
    for seed in $seeds; do
        python train_energy.py --size_factor_sampled_ood $factor --seed $seed --id cifar10 --ood ti_300k --gpu_idx 0
    done
done