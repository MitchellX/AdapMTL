# resnet34 baseline
python baseline.py --architecture resnet34 --save_dir ./outputs/resnet34_1028/baseline/
# load from the checkpoint
CUDA_VISIBLE_DEVICES=1 python baseline.py --architecture resnet34 --save_dir ./outputs/resnet34_1028/baseline/ --reload outputs/resnet34_1028/baseline/segment_semantic_normal_depth_zbuffer_18999.model

# resnet34 pruning
python pruning.py --prune_ratio 0.85 --architecture resnet34 --save_dir outputs/resnet34_1028/pruning/ --load_checkpoint outputs/resnet34_1028/baseline/best_segment_semantic_normal_depth_zbuffer.model

# resnet34 finetune
CUDA_VISIBLE_DEVICES=1 python finetune.py --prune_ratio 0.97

# mobilenet pruning
python pruning.py --prune_ratio ${ratio} --architecture mobilenetv2 \
--load_checkpoint outputs/mobilenet_1030/baseline_15k/best_segment_semantic_normal_depth_zbuffer.model \
--save_dir outputs/mobilenet_1030/pruning/


# mobilenet finetune
python baseline.py \
    --sInit_value -18 \
    --weight_decay 0.00000005 \
    --ratio 0.5 \
    --save_dir /home/mingcanxiang_umass_edu/work_path/TreeMTL/mobilenetv2_0328/minus18_decay0.00000005/ \
    --batch_size 16 \
    --iters 20000 \
    --architecture mobilenetv2

# mobilement training from dense model
python baseline.py
    --sInit_value -10 \
    --weight_decay 0.0001 \
    --ratio 0.6 \
    --reload outputs/unity/dense_mobilnetv2_decay_0.0001/best_segment_semantic_normal_depth_zbuffer.model \
    --save_dir outputs/unity/mobilenetv2_0401/minus10_decay0.0001_0.6/ \
    --architecture mobilenetv2

CUDA_VISIBLE_DEVICES=1 python baseline.py --reload outputs/unity/dense_mobilnetv2_decay_0.0001/best_segment_semantic_normal_depth_zbuffer.model --architecture mobilenetv2 --save_dir outputs/unity/mobilenetv2_0410/minus100_decay0.0001_0.9 --sInit_value -100 --ratio 0.9

python baseline_taskonomy.py \
    --dataset taskonomy \
    --architecture mobilenetv2 \
    --iters 100000 \
    --lr 0.0001 \
    --decay_lr_freq 12000 \
    --decay_lr_rate 0.3 \
    --sInit_value -1000 \
    --ratio 0.9 \
    --save_dir outputs/unity/taskonomy_dense_mobilenetv2_decay_0.0001/