for bk_sp in 0.6 0.99; do
    for ratio in 0.9 0.93 0.95 0.97 0.99;do
        python pruning_head.py \
        --backbone_sparsity ${bk_sp}\
        --prune_ratio ${ratio} \
        --save_dir ./outputs/unity/resnet34_${bk_sp}/pruning/ \
        --load_checkpoint ./outputs/resnet34_1119/baseline/best_segment_semantic_normal_depth_zbuffer.model \
        --architecture resnet34
    done
done
