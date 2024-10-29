from main.auto_models import MTSeqBackbone
from main.algorithms import enumerator
from main.algorithms import reorg_two_task_results, compute_weights, metric_inference

from torch.utils.data import DataLoader

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics
from models.Deeplab_ResNet34 import Deeplab_ResNet_Backbone

import torch
from main.auto_models import MTSeqModel
from main.algorithms import coarse_to_fined
from models.mobilenetv2 import mobilenetv2
from main.tester import Tester

import os
import argparse
from main.utils import logger_init, logging
from args import args

logger_init(f"{args.architecture}_baseline", log_dir=args.save_dir)

# change the network or dataset
if args.architecture == 'mobilenetv2':
    prototxt = 'models/mobilenetv2.prototxt'
    mapping = {0:[0,1,2,3,4,5,6], 1:[7,8,9,10,11,12,13,14,15,16,17], 2:[18,19,20,21,22],
            3:[23,24,25,26,27,28,29,30], 4:[31], 5:[32]} # mapping for MobileNetV2
    backbone = MTSeqBackbone(prototxt)
    B = len(backbone.basic_blocks)
    # real backbone
    backbone = mobilenetv2()
else:
    prototxt = 'models/deeplab_resnet34_adashare.prototxt'
    # mapping for resnet34
    mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
    backbone = MTSeqBackbone(prototxt)
    B = len(backbone.basic_blocks)
    # real backbone
    backbone = Deeplab_ResNet_Backbone()

T = 3 # NYUv2 has 3 tasks, Taskonomy has 5 tasks
coarse_B = 5
layout_list = enumerator(T, coarse_B)

# =============== prepare the dataset  ===============
dataroot = '/home/tongping/dataset/taskonomy/' # Your root
# dataroot = '/home/mingcanxiang_umass_edu/work_path/taskonomy/' # Your root

criterionDict = {}
metricDict = {}

tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}

dataset = Taskonomy(dataroot, 'train', crop_h=224, crop_w=224)
trainDataloader = DataLoader(dataset, batch_size=16, shuffle=True)

dataset = Taskonomy(dataroot, 'test', crop_h=224, crop_w=224)
valDataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for task in tasks:
    criterionDict[task] = TaskonomyCriterions(task, dataroot)
    metricDict[task] = TaskonomyMetrics(task, dataroot)


# coarse_layout = layout_list[45]

# all shared multi-task model
coarse_layout = layout_list[0]
fined_layout = coarse_to_fined(coarse_layout, B, mapping)

# feature_dim = backbone(torch.rand(2,3,224,224)).shape[1]
feature_dim = backbone(torch.ones(16, 3, 321, 321)).shape[1]
model = MTSeqModel(prototxt, backbone, layout=fined_layout, feature_dim=feature_dim, cls_num=cls_num)

tester = Tester(model.cuda(), tasks, trainDataloader, valDataloader, criterionDict, metricDict, architecture=args.architecture, logging=logging)
tester.test(reload=args.path, priority=[1, 1, 1, 1, 1])       # load from savePath + reload