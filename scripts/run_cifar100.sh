set -e
set -x

CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 100 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --nce_temp 0.5 \
    --sparsity_weight 0.6 \
    --weight_decay 5e-5 \
    --reg_weight 0.5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name cifar100