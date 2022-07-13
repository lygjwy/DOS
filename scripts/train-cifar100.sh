# Train CIFAR100 CLF
## random sampling
python train_rand_resa.py --id cifar100 --ood tiny_images

## confidence sampling
python train_conf_resa.py --id cifar100 --ood tiny_images --ood_quantile 0.5
