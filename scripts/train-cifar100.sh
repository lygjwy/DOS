# Train CIFAR100 CLF
## regular training
# python train.py --dataset cifar100

# single gpu
## random sampling
python train_unsa_rand.py --id cifar100 --training uni --beta 0.5 --output_dir ep0 --scheduler lambda --parallel
python train_unsa_rand.py --id cifar100 --training abs --beta 1.0 --output_dir ep0 --scheduler multistep --parallel
python train_unsa_rand.py --id cifar100 --training binary --beta 2.0 --include_binary --output_dir ep0 --scheduler lambda --parallel

## quantile sampling
python train_unsa_quantile.py --id cifar100 --training uni --weighting msp --beta 0.5 --output_dir ep0 --ood_quantile 0.5 --scheduler lambda --parallel
python train_unsa_quantile.py --id cifar100 --training abs --weighting abs --beta 1.0 --output_dir ep0 --ood_quantile 0.5 --scheduler multistep --parallel
python train_unsa_quantile.py --id cifar100 --training binary --weighting energy --beta 2.0 --include_binary --output_dir ep0 --ood_quantile 0.5 --scheduler lambda --parallel

## ours sampling
python train_unsa_bidir.py --id cifar100 --training binary --beta 2.0 --include_binary --output_dir ep0 --scheduler lambda --energy_ratio 0.5 --kl_ratio 0.5 --parallel
python train_unsa_bidir.py --id cifar100 --training binary --beta 2.0 --include_binary --output_dir ep0 --scheduler lambda --energy_ratio 0.25 --kl_ratio 0.75 --parallel