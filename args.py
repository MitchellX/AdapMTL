import argparse

parser = argparse.ArgumentParser(description="multitask training")

# parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--batch_size', default=16, type=int)
# parser.add_argument('--dataroot', default='/home/tongping/dataset/taskonomy/', type=str)
parser.add_argument('--save_dir', default='./outputs/unity/xmc_test/', type=str)
# parser.add_argument('--save_dir', default='./outputs/resnet34_0208/xmc_test/', type=str)
parser.add_argument('--architecture', default='resnet34', type=str, choices=["mobilenetv2", "resnet34"])
parser.add_argument('--reload', default=None, type=str)
parser.add_argument('--sInit_type', default='constant', type=str)
parser.add_argument('--sInit_value', default=-30.5, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--ratio', default=0.9, type=float)
parser.add_argument('--sparse_function', default='sigmoid', type=str)
parser.add_argument('--conv_type', default='STRConv', type=str)

# first choices for NYUv2, second choices for taskonomy
parser.add_argument('--dataset', default='NYUV2', type=str, choices=["NYUV2", "taskonomy"])
parser.add_argument('--iters', default=20000, type=int, choices=[20000, 100000])
parser.add_argument('--lr', default=0.001, type=float, choices=[0.001, 0.0001])
parser.add_argument('--decay_lr_freq', default=4000, type=int, choices=[4000, 12000])
parser.add_argument('--decay_lr_rate', default=0.5, type=float, choices=[0.5, 0.3])
parser.add_argument('--val_iters', default=5000, type=int, choices=[200, 5000])

# for testing purposes
parser.add_argument('--save_log_dir', default='logs', type=str)
parser.add_argument('--log_name', default='logs', type=str)
parser.add_argument('--path', default='/home/tongping/mingcan/github/Pruning/TreeMTL/outputs/resnet34_0208/baseline_minus20_decay0.0005/baseline_minus20_decay0.0005best_segment_semantic_normal_depth_zbuffer.model', type=str)


args = parser.parse_args()
