for ratio in 0.955 0.9955;do
    python finetune.py \
    --prune_ratio ${ratio} \
    --architecture mobilenetv2 \
    --save_dir ./outputs/unity/mobilenet_0.9
done