for ratio in 0.9 0.93 0.95 0.97 0.99;do
    python finetune.py \
    --prune_ratio ${ratio} \
    --architecture resnet34 \
    --save_dir ./outputs/unity/resnet34_0.99
done