output_path=outputs/unity/mobilenetv2_0328/
for ratio in `ls ${output_path}`;do
    python test.py \
    --path ${output_path}/${ratio}/best_segment_semantic_normal_depth_zbuffer.model \
    --architecture mobilenetv2 \
    --save_dir logs \
    --log_name mobilenetv2_accuracy_saprsity
done