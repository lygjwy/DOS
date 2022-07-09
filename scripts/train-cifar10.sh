# Train CIFAR10 CLF

# random sampling
python train_rand_resa.py --id cifar10 --ood tiny_images --lr 0.1 --epochs 100 --gpu_idx 0

# weighted resampling
for w in conf dens; do
    python train_weighted_resa.py --id cifar10 --ood tiny_images --weight $w --lr 0.1 --epochs 100 --gpu_idx 0
    python train_weighted_resa.py --id cifar10 --ood tiny_images --weight $w --cond_sample --lr 0.1 --epochs 100 --gpu_idx 0
done