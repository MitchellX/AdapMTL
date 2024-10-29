# for bk_sp in 0.9; do
#     for ratio in 0.955 0.9955;do
#         python pruning_head.py \
#         --backbone_sparsity ${bk_sp}\
#         --prune_ratio ${ratio} \
#         --save_dir outputs/unity/mobilenet_${bk_sp}/pruning/ \
#         --architecture mobilenetv2 \
#         --load_checkpoint outputs/mobilenet_1113/baseline_15k/best_segment_semantic_normal_depth_zbuffer.model
#     done
# done

# for bk_sp in 0.5 0.6 0.7 0.8 0.9 0.98; do
#     for ratio in 0.9 0.93 0.95 0.97 0.99;do
#         echo "python pruning_head.py --prune_ratio ${ratio}"
#         python pruning_head.py \
#         --backbone_sparsity ${bk_sp}\
#         --prune_ratio ${ratio} \
#         --save_dir outputs/mobilenet_${bk_sp}/pruning/
#     done
# done


# resnet34
output_path=outputs/unity/resnet34_accuracy_sparsity/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture resnet34 \
    --save_dir logs \
    --log_name resnet34_accuracy_saprsity
done

output_path=outputs/unity/resnet34_0321/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture resnet34 \
    --save_dir logs \
    --log_name resnet34_accuracy_saprsity
done

output_path=outputs/unity/resnet34_0316/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture resnet34 \
    --save_dir logs \
    --log_name resnet34_accuracy_saprsity
done

output_path=outputs/unity/resnet34_0308/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture resnet34 \
    --save_dir logs \
    --log_name resnet34_accuracy_saprsity
done

output_path=outputs/unity/resnet34_0305/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture resnet34 \
    --save_dir logs \
    --log_name resnet34_accuracy_saprsity
done

output_path=outputs/unity/resnet34_0302/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture resnet34 \
    --save_dir logs \
    --log_name resnet34_accuracy_saprsity
done