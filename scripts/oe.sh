# retrain classifier from scratch
python oe.py --id cifar10 --ood tiny_images --lr 0.1 --epochs 100 --gpu_idx 0
python oe.py --id cifar100 --ood tiny_images --lr 0.1 --epochs 100 --gpu_idx 1

# finetune classifier
python oe.py --id cifar10 --ood tiny_images --pretrain /home/hzheng/jwy/resampling/outputs/cifar10/wrn40/cla_best.pth --lr 0.001 --epochs 10 --gpu_idx 0
python oe.py --id cifar100 --ood tiny_images --pretrain /home/hzheng/jwy/resampling/outputs/cifar100/wrn40/cla_best.pth --lr 0.001 --epochs 10 --gpu_idx 1