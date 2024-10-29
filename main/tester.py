import imp
from pyexpat import model
from statistics import mode
from tkinter.messagebox import NO
import numpy as np
from sys import exit
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import logging
from args import args
from models.Deeplab_ResNet34 import STRConv

class Tester():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict, 
                 architecture = 'resnet', logging=None,
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 print_iters=50, val_iters=200, save_iters=200,
                 early_stop=False, stop=3, good_metric=10):
        super(Tester, self).__init__()
        self.model = model
        self.startIter = 0
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
#         self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        self.architecture = architecture

        self.tasks = tasks
        
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.criterion_dict = criterion_dict
        self.metric_dict = metric_dict
        
        self.loss_list = {}
        # self.set_train_loss()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.save_iters = save_iters
        
        self.early_stop = early_stop
        if self.early_stop:
            self.counter = 0
            self.stop = stop # Define how many consencutive good validate results we need
            self.good_metric = good_metric # Define at least how many good validate metrics every time to make counter+1
    
    def test(self, savePath=None, reload=None, priority=None):
        logging.info('******* path: {}'.format(reload))
        if reload is not None:
            self.load_model(reload)
        else:
            logging.info("use default model")

        # assign the priority for different tasks
        loss_priority = {}
        if priority is None:
            loss_priority = {task: 1 for task in self.tasks}
        else:
            for i in range(len(self.tasks)):
                loss_priority[self.tasks[i]] = priority[i]
        logging.info(loss_priority)

        # this is the test function
        self.validate(it=self.startIter, loss_priority=loss_priority)
        return

    # calculate the whole numbers of the weights and non-zero weights
    def check_parameters(self, check_model, name):
        total = 0
        total_nonzero = 0
        total_conv = 0
        conv_nonzero = 0

        # calculate the whole numbers of the weights
        # currently, I only prune the backbone
        for m in check_model.modules():
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

        logging.info('{}:'.format(name))
        logging.info('total_conv: {}'.format(total_conv))
        logging.info('conv_nonzero: {}'.format(conv_nonzero))
        logging.info('sparsity: {}'.format(conv_nonzero / total_conv))
        logging.info('======================================================================')

    # Storing sparsity and threshold statistics for STRConv models
    def store_sparsity(self, check_model, writer, iter, name):

        # Storing sparsity and threshold statistics for STRConv models
        if args.conv_type == "STRConv":
            count = 0
            sum_sparse = 0.0
            for n, m in check_model.named_modules():
                if isinstance(m, STRConv):
                    sparsity, total_params, thresh = m.getSparsity()
                    # writer.add_scalar("Layer_sparsity/{}_{}".format(name, n), sparsity, iter)
                    # writer.add_scalar("Layer_thresh/{}_{}".format(name, n), thresh, iter)
                    sum_sparse += int(((100 - sparsity) / 100) * total_params)
                    count += total_params
            total_sparsity = 100 - (100 * sum_sparse / count)
            # writer.add_scalar("sparsity/{}".format(name), total_sparsity, iter)

            return total_sparsity, count
        # ======================================================================

    def validate(self, it=0, loss_priority=None, writer=None):
        self.model.eval()

        # calculate the whole numbers of the weights and non-zero weights
        # self.check_parameters(self.model.backbone, 'backbone')
        # self.check_parameters(self.model.heads['segment_semantic'], 'segment_semantic head')
        # self.check_parameters(self.model.heads['normal'], 'normal head')
        # self.check_parameters(self.model.heads['depth_zbuffer'], 'depth_zbuffer head')

        s1, t1 = self.store_sparsity(self.model.backbone, writer, it, 'Backbone')
        s2, t2 = self.store_sparsity(self.model.heads['segment_semantic'], writer, it, 'Seg')
        s3, t3 = self.store_sparsity(self.model.heads['normal'], writer, it, 'Nor')
        s4, t4 = self.store_sparsity(self.model.heads['depth_zbuffer'], writer, it, 'Dep')

        overall_sparsity = (s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4) / 100.0 / (t1 + t2 + t3 + t4)

        logging.info('Backbone sparsity: {}'.format(round(s1, 2)))
        logging.info('segment_semantic sparsity: {}'.format(round(s2, 2)))
        logging.info('normal sparsity: {}'.format(round(s3, 2)))
        logging.info('depth sparsity: {}'.format(round(s4, 2)))
        logging.info('overall_sparsity: {}'.format(round(overall_sparsity * 100, 2)))
        logging.info('======================================================================')

        loss_list = {}
        for task in self.tasks:
            loss_list[task] = []
        
        for i, data in enumerate(self.val_dataloader):
            x = data['input'].cuda()
            output = self.model(x)

            for task in self.tasks:
                y = data[task].cuda()
                if task + '_mask' in data:
                    tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
                    self.metric_dict[task](output[task], y, data[task + '_mask'].cuda())
                else:
                    tloss = self.criterion_dict[task](output[task], y)
                    self.metric_dict[task](output[task], y)
                loss_list[task].append(tloss.item())
        
        task_val_results = {}
        for task in self.tasks:
            avg_loss = np.mean(loss_list[task])
            val_results = self.metric_dict[task].val_metrics()
            if writer != None:
                writer.add_scalar('Loss/val/' + task, avg_loss, it)
                for metric in val_results:
                    writer.add_scalar('Metric/' + task + '/' + metric, val_results[metric], it)
            # if self.early_stop:
            task_val_results[task] = val_results
            # print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
            logging.info('[Iter {} Task {}] Val Loss: {:.4f}'.format(it, task[:4], avg_loss))
            logging.info(val_results)
        if self.early_stop:
            self.early_stop_monitor(task_val_results)
        logging.info('======================================================================')

        # evaluation for different dataset
        if args.dataset == "taskonomy":
            test_error = 10
            err = task_val_results['segment_semantic']['err']
            cosine_similarity = task_val_results['normal']['cosine_similarity']
            abs_err = task_val_results['depth_zbuffer']['abs_err']
            key_err = task_val_results['keypoints2d']['key_err']
            edge_err = task_val_results['edge_texture']['edge_err']

            if self.architecture == 'resnet34':
                test_error = err / 0.5168 + cosine_similarity / 0.8745 + abs_err / 0.0195 + key_err / 0.2003 + edge_err / 0.2082
            else:        # mobilenetv2
                test_error = err / 0.9770 + cosine_similarity / 0.7625 + abs_err / 0.0277 + key_err / 0.2232 + edge_err / 0.2519
            
            logging.info("average_error: {}".format(round(test_error, 4)))

        elif args.dataset == "NYUV2":
            # test the metrics
            test_acc = 0
            mIoU, Pixel_acc = task_val_results['segment_semantic']['mIoU'], task_val_results['segment_semantic']['Pixel Acc']
            angle_mean, angle_median, angle1, angle2, angle3 = task_val_results['normal']['Angle Mean'], task_val_results['normal']['Angle Median'], task_val_results['normal']['Angle 11.25'], task_val_results['normal']['Angle 22.5'], task_val_results['normal']['Angle 30']
            abs_err, rel_err, sigma1, sigma2, sigma3 = task_val_results['depth_zbuffer']['abs_err'], task_val_results['depth_zbuffer']['rel_err'], task_val_results['depth_zbuffer']['sigma_1.25'], task_val_results['depth_zbuffer']['sigma_1.25^2'], task_val_results['depth_zbuffer']['sigma_1.25^3']
            
            '''
            # independent model
            if self.architecture == 'resnet34':
                seg1, seg2 = 0.2650, 0.5820
                norm1, norm2, norm3, norm_mean, norm_median = 29.40, 72.30, 87.30, 17.70, 16.30
                dep1, dep2, dep3, dep_abs, dep_rel = 57.80, 85.80, 96.00, 0.62, 0.24
            else:        # mobilenetv2
                seg1, seg2 = 0.2036, 0.4944
                norm1, norm2, norm3, norm_mean, norm_median = 28.37, 70.20, 85.58, 18.17, 16.62
                dep1, dep2, dep3, dep_abs, dep_rel = 47.92, 78.46, 92.81, 0.77, 0.28
            '''

            if self.architecture == 'resnet34':
                seg1, seg2 = 0.2525, 0.5773
                norm_mean, norm_median, norm1, norm2, norm3  = 17.2398, 14.9797, 36.433, 72.0786, 85.2728
                dep_abs, dep_rel, dep1, dep2, dep3 = 0.5551, 0.2151, 64.4659, 89.8674, 97.4203
            else:        # mobilenetv2
                seg1, seg2 = 0.1849, 0.4864
                norm_mean, norm_median, norm1, norm2, norm3  = 17.7522, 16.2718, 28.5603, 73.9923, 87.3249
                dep_abs, dep_rel, dep1, dep2, dep3 = 0.587, 0.2404, 61.9075, 87.4701, 96.1797
            
            sum = loss_priority['segment_semantic'] + loss_priority['normal'] + loss_priority['depth_zbuffer']

            # # [8 metrics] the denominator of each metric is the independent model performance score
            # test_acc = loss_priority['segment_semantic'] / sum * 0.5 * (mIoU/seg1 + Pixel_acc/seg2) + \
            #             loss_priority['normal'] / sum * 0.33 * (angle1/norm1 + angle2/norm2 + angle3/norm3) + \
            #             loss_priority['depth_zbuffer'] / sum * 0.33 * (sigma1/dep1 + sigma2/dep2 + sigma3/dep3)
            # 12 metrics
            test_acc = loss_priority['segment_semantic'] / sum * 0.5 * (mIoU/seg1 + Pixel_acc/seg2) + \
                        loss_priority['normal'] / sum * (angle1/norm1 + angle2/norm2 + angle3/norm3 - angle_mean/norm_mean - angle_median/norm_median) + \
                        loss_priority['depth_zbuffer'] / sum * (sigma1/dep1 + sigma2/dep2 + sigma3/dep3 - abs_err/dep_abs - rel_err/dep_rel)
            logging.info("test score: {}".format(round(test_acc, 4)))
    
    
    def load_model(self, reload):
        model_name = True
        for task in self.tasks:
            if task not in reload:
                model_name = False
                break
        if model_name:
            state = torch.load(reload)
            self.startIter = state['iter']
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            print('Cannot load from models trained from different tasks.')
            exit()
        return
    
    def save_model(self, it, savePath):
        state = {'iter': it,
                'state_dict': self.model.state_dict(),
                'layout': self.model.layout,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
        if hasattr(self.model, 'branch') and self.model.branch is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '_b' + str(self.model.branch) + '.model')
        elif hasattr(self.model, 'layout') and self.model.layout is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '_' + str(it) + '.model')
        return
    
    def print_train_loss(self, it, writer=None):
        # Function: Print loss for each task
        for task in self.tasks:
            if self.loss_list[task]:
                avg_loss = np.mean(self.loss_list[task])
            else:
                continue
            if writer != None:
                writer.add_scalar('Loss/train/' + task, avg_loss, it)
            print('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), np.mean(self.loss_list['total'])), flush=True)
        print('======================================================================', flush=True)
        return
