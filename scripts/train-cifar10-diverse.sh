groups='1 2 4 6'

for group in $groups; do
    python train_diverse.py --num_group $group --gpu_idx 0
done