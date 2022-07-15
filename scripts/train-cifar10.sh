# Train CIFAR10 CLF
## regular training
# python train.py --dataset cifar10

## random sampling
# python train_rand_resa.py --id cifar10 --ood tiny_images

## confidence sampling
python train_conf_resa.py --id cifar10 --ood tiny_images --ood_quantile 0.0
python train_conf_resa.py --id cifar10 --ood tiny_images --ood_quantile 0.125

## density sampling
python train_dens_resa.py --id cifar10 --ood tiny_images --ood_quantile 0.0
python train_dens_resa.py --id cifar10 --ood tiny_images --ood_quantile 0.125