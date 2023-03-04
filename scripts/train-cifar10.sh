# Train CIFAR10 CLF: tune loss regulating coefficient with different seeds

seeds='3 14 42'

for seed in $seeds; do
    python train_dual_batch.py --seed $seed --id cifar10 --ood ti_300k --ood_ratio 0.9 --gpu_idx 0
done