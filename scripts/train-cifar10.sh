# Train CIFAR10 CLF

# random resampling
# for t in fix_loc var_loc fix_rand var_rand; do
#     # tuning
#     python train_rand_resa.py --id cifar10 --ood tiny_images --pretrain ./outputs/cifar10/wrn40/cla_best.pth --training $t --lr 0.001 --epochs 10 --gpu_idx 0
#     # retraining
#     python train_rand_resa.py --id cifar10 --ood tiny_images --training $t --lr 0.1 --epochs 100 --gpu_idx 0
# done

# classification resampling
for w in prob logit; do
    for t in fix var; do
        for r in True False; do
            # tuning
            python train_cla_resa.py --id cifar10 --ood tiny_images --training $t --weight_type $w --replacement $r --pretrain ./outputs/cifar10/wrn40/cla_best.pth --lr 0.001 --epochs 10 --gpu_idx 0
            # retraining
            python train_cla_resa.py --id cifar10 --ood tiny_images --training $t --weight_type $w --replacement $r --lr 0.1 --epochs 100 --gpu_idx 0
        done
    done
done