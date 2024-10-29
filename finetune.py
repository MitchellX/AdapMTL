from main.auto_models import MTSeqBackbone
from main.algorithms import enumerator
from main.algorithms import reorg_two_task_results, compute_weights, metric_inference

from torch.utils.data import DataLoader

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics

import torch
from main.auto_models import MTSeqModel
from main.algorithms import coarse_to_fined
from main.trainer import Trainer

import argparse
import os
from main.utils import logger_init, logging


parser = argparse.ArgumentParser(description="multitask pruning")

# parser.add_argument('--architecture', default='mobilenetv2', type=str)
# parser.add_argument('--save_dir', default='outputs/all_shared_mobilenetv2/0928_prune_head/', type=str)

# batch size=10, 16. iters=32k, 20k
parser.add_argument('--iters', default=32000, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--prune_ratio', default=0.99, type=float)
parser.add_argument('--architecture', default='resnet34', type=str)
parser.add_argument('--save_dir', default='outputs/unity/resnet34_0.6', type=str)
parser.add_argument('--reload', default=None, type=str)
parser.add_argument('--priority', default=None, nargs='+', type=int)
# parser.add_argument('--priority', default=[8,1,1], type=list)

args = parser.parse_args()

# change the network or dataset
if args.architecture == 'mobilenetv2':
    prototxt = 'models/mobilenetv2.prototxt'
    mapping = {0:[0,1,2,3,4,5,6], 1:[7,8,9,10,11,12,13,14,15,16,17], 2:[18,19,20,21,22],
            3:[23,24,25,26,27,28,29,30], 4:[31], 5:[32]} # mapping for MobileNetV2
else:
    prototxt = 'models/deeplab_resnet34_adashare.prototxt'
    # mapping for resnet34
    mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
backbone = MTSeqBackbone(prototxt)
B = len(backbone.basic_blocks)

T = 3 # NYUv2 has 3 tasks, Taskonomy has 5 tasks
coarse_B = 5
layout_list = enumerator(T, coarse_B)

# =============== prepare the dataset  ===============
# dataroot = '/home/mingcanxiang_umass_edu/work_path/nyu_v2' # Your root
dataroot = 'data/NYUv2' # Your root

criterionDict = {}
metricDict = {}

tasks = ['segment_semantic','normal','depth_zbuffer']
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
trainDataloader = DataLoader(dataset, args.batch_size, shuffle=True)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
valDataloader = DataLoader(dataset, args.batch_size, shuffle=True)

for task in tasks:
    criterionDict[task] = NYUCriterions(task)
    metricDict[task] = NYUMetrics(task)


coarse_layout = layout_list[45]

# all shared multi-task model
coarse_layout = layout_list[0]
fined_layout = coarse_to_fined(coarse_layout, B, mapping)

feature_dim = backbone(torch.rand(1,3,224,224)).shape[1]
model = MTSeqModel(prototxt, layout=fined_layout, feature_dim=feature_dim, cls_num=cls_num)

if args.priority:
    save_path = args.save_dir + '/finetune/{}_priority_{}_{}_{}/'.format(args.prune_ratio, args.priority[0], args.priority[1], args.priority[2])
else:
    save_path = args.save_dir + '/finetune/{}/'.format(args.prune_ratio)

pruned_model_path = args.save_dir + '/pruning/{0}/segment_semantic_normal_depth_zbuffer_{0}.model'.format(args.prune_ratio)

logger_init(f"{args.architecture}_finetune", log_dir=save_path)

trainer = Trainer(model.cuda(), tasks, trainDataloader, valDataloader, criterionDict, metricDict, architecture=args.architecture, logging=logging)
trainer.train(iters=args.iters, priority=args.priority, savePath=save_path, load_pruned=pruned_model_path, reload=args.reload, writerPath=save_path)