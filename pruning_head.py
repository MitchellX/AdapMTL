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
from main.tester import Tester

import torch.nn as nn
import argparse
import os

from main.utils import logger_init, logging

parser = argparse.ArgumentParser(description="multitask pruning")

# MobileNetv2
parser.add_argument('--load_checkpoint', default='outputs/mobilenet_1113/baseline_15k/best_segment_semantic_normal_depth_zbuffer.model', type=str)
parser.add_argument('--save_dir', default='outputs/mobilenet_1125/pruning', type=str)
parser.add_argument('--architecture', default='mobilenetv2', type=str)

# ResNet34
# parser.add_argument('--load_checkpoint', default='outputs/resnet34_1119/baseline/best_segment_semantic_normal_depth_zbuffer.model', type=str)
# parser.add_argument('--save_dir', default='./outputs/resnet34_1121/pruning/', type=str)
# parser.add_argument('--architecture', default='resnet34', type=str)

parser.add_argument('--conv_only', default='Ture',
                    help='prune the conv layers only')
parser.add_argument('--prune_ratio', default=0.9, type=float)
parser.add_argument('--backbone_sparsity', default=0.9, type=float)

# polish pruning.py 以适应以后加入priority

args = parser.parse_args()

logger_init("prune_{}".format(str(args.prune_ratio)), log_dir=os.path.join(args.save_dir, str(args.prune_ratio)))

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

# =============== prepare the dataset  ================
dataroot = 'data/NYUv2' # Your root
# dataroot = '/home/mingcanxiang_umass_edu/work_path/nyu_v2' # Your root

criterionDict = {}
metricDict = {}

tasks = ['segment_semantic','normal','depth_zbuffer']
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
trainDataloader = DataLoader(dataset, 16, shuffle=True)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
valDataloader = DataLoader(dataset, 16, shuffle=True)

for task in tasks:
    criterionDict[task] = NYUCriterions(task)
    metricDict[task] = NYUMetrics(task)


# =============== prepare the multitask model  ================
coarse_layout = layout_list[45]

# all shared multi-task model
coarse_layout = layout_list[0]
fined_layout = coarse_to_fined(coarse_layout, B, mapping)

feature_dim = backbone(torch.rand(1,3,224,224)).shape[1]
model = MTSeqModel(prototxt, layout=fined_layout, feature_dim=feature_dim, cls_num=cls_num)

model.cuda()


# # =============== test before pruning  ================
# you must use the tester, becuase you need to load the parameters from the best model
tester = Tester(model, tasks, trainDataloader, valDataloader, criterionDict, metricDict, logging=logging)
tester.test(reload=args.load_checkpoint)       # load from savePath + reload


# ************** pruning function **************
def pruning(whole_models, prune_ratio):
    total = 0
    total_nonzero = 0
    linear = 0
    linear_nonzero = 0
    total_conv = 0
    conv_nonzero = 0

    # calculate the whole numbers of the weights
    for model in whole_models:
        for m in model.modules():
            # ========== when calculating the total number of parameters, we count into the linear layer ============
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if isinstance(m, nn.Conv2d):
                    total_conv += m.weight.data.numel()
                    total += m.weight.data.numel()
                    mask = m.weight.data.abs().clone().gt(0).float().cuda()     # non-zero values
                    total_nonzero += torch.sum(mask)
                    conv_nonzero += torch.sum(mask)
                else:
                    total += m.weight.data.numel()
                    mask = m.weight.data.abs().clone().gt(0).float().cuda()
                    total_nonzero += torch.sum(mask)
                    # here I use mask for not counting the zero values repeatedly

    # use conv_weight to store the stretched weight
    conv_weights = torch.zeros(total_conv)

    index = 0
    for model in whole_models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                index += size

    # ========== while applying the LT magnitude-based pruning, we only prune the weights of conv layers ============
    # ========== and make the global remained ratio = 1 - p = 1 - --prune_ratio ============

    y, i = torch.sort(conv_weights)
    if args.conv_only:
        thre_index = total_conv - conv_nonzero + int(conv_nonzero * prune_ratio)
    else:
        thre_index = total - total_nonzero + int(total_nonzero * prune_ratio)

    # if any weights are less than the threshold, then make them to zero values.
    thre = y[int(thre_index)]
    pruned = 0
    logging.info('Pruning threshold: {}'.format(thre))

    zero_flag = False
    layer = 0

    for model in whole_models:
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                layer = layer + 1
                weight_copy = m.weight.data.abs().clone()
                # use mask to mark the values greater than threshold
                mask = weight_copy.gt(thre).float().cuda()              # xmc wait change1: float or int
                if isinstance(m, nn.Linear):
                    # TYPE = 'Linear'
                    TYPE = 'FC'
                    mask = weight_copy.gt(0).float().cuda()     # currently, we don't care about FC layers
                else:
                    TYPE = 'Conv'
                
                # calculate the pruned weights
                pruned += (mask.numel() - torch.sum(mask))

                # if any weights are less than the threshold, then make them to zero values.
                # apply the mask to layer weights
                m.weight.data.mul_(mask)

                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                logging.info('layer index: {:d} ({:s}) \t total params: {:d} \t remaining params: {:d} \t remained ratio: {:2f}'.
                        format(layer, TYPE, mask.numel(), int(torch.sum(mask)), (torch.sum(mask) / mask.numel()).float()))
                logging.info('layer index: {:d} ({:s}) \t total params: {:d} \t remaining params: {:d} \t remained ratio: {:2f}'.
                        format(layer, TYPE, mask.numel(), int(torch.sum(mask)), (torch.sum(mask) / mask.numel()).float()))

    logging.info('Total params: {}, Pruned params: {}, Pruned ratio: {} \n'.format(total, pruned, pruned / total))
    logging.info('Total params: {}, Pruned params: {}, Pruned ratio: {} \n'.format(total, pruned, pruned / total))


# ************************** pruning function  ****************************
os.makedirs(os.path.join(args.save_dir, str(args.prune_ratio)), exist_ok=True)
# pruning([model.backbone], 0.9)  # should be 0.972
pruning([model.backbone], args.backbone_sparsity)  # should be 0.972
pruning([model.heads['segment_semantic'], model.heads['normal'], model.heads['depth_zbuffer']], args.prune_ratio)

# =============== test after pruning  ================
tester = Tester(model, tasks, trainDataloader, valDataloader, criterionDict, metricDict)
tester.test()       # use the default model

# save the model
tester.save_model(str(args.prune_ratio), os.path.join(args.save_dir, str(args.prune_ratio)) + '/')
logging.info("Successfully, saved the model")