import imp
import logging
import numpy as np
from sys import exit
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# add a better optimizer
from main.pcgrad import PCGrad

best_acc = -10

class Trainer():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict,
                 architecture = 'resnet', logging=None,
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 print_iters=50, val_iters=200, save_iters=500,
                 early_stop=False, stop=3, good_metric=10):
        super(Trainer, self).__init__()
        self.model = model
        self.startIter = 0

        # lambda function: filter all the layers and keep the layers who require gradients.
        # original optimizer
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # replace with a better optimizer
        self.optimizer = PCGrad(torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001))
#         self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer.optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        self.architecture = architecture
        self.tasks = tasks
        
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.criterion_dict = criterion_dict
        self.metric_dict = metric_dict
        
        self.loss_list = {}
        self.set_train_loss()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.save_iters = save_iters
        
        self.early_stop = early_stop
        if self.early_stop:
            self.counter = 0
            self.stop = stop # Define how many consencutive good validate results we need
            self.good_metric = good_metric # Define at least how many good validate metrics every time to make counter+1
    
    def train(self, iters, priority=None, savePath=None, load_pruned=None, reload=None, writerPath=None):
        global best_acc

        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath)
        else:
            writer = None
        
        self.model.train()

        # load from the pruned model
        if load_pruned is not None and savePath is not None and reload is None:
            self.load_model(savePath, load_pruned)

        # load from the checkpoint
        if reload is not None and savePath is not None:
            self.load_model(savePath, reload)

        # assign the priority for different tasks
        loss_priority = {}
        if priority is None:
            loss_priority = {task: 1 for task in self.tasks}
        else:
            for i in range(len(self.tasks)):
                loss_priority[self.tasks[i]] = priority[i]
        logging.info(loss_priority)

        for i in range(self.startIter, iters):
            if self.early_stop and self.counter >= self.stop:
                if savePath is not None:
                    self.save_model(i, savePath)
                logging.info('Early Stop Occur at {} Iter'.format((i+1)))
                break
            
            self.train_step(loss_priority)

            # Temporarily, for test
            # self.print_iters, self.val_iters, self.save_iters = 1, 1, 1

            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i, writer)
                self.set_train_loss()       # clear the training loss
            if (i+1) % self.val_iters == 0:
                self.validate(i, writer, savePath, loss_priority)
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    self.save_model(i, savePath)
            
        # Reset loss list and the data iters
        self.set_train_loss()
        return
    
    def train_step(self, loss_priority):
        self.model.train()
        try:
            data = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data = next(self.train_iter)
            
        x = data['input'].cuda()
        self.optimizer.zero_grad()
        output = self.model(x)
         
         # ============================ rewrite the loss format to customize the better optimizer ==========================================
        loss = 0
        tloss = {}
        for task in self.tasks:
            y = data[task].cuda()       # y is the ground truth
            if task + '_mask' in data:
                tloss[task] = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
            else:
                tloss[task] = self.criterion_dict[task](output[task], y)
                
            self.loss_list[task].append(tloss[task].item())
            loss += loss_priority[task] * tloss[task]
        self.loss_list['total'].append(loss.item())
        
        self.optimizer.pc_backward([ loss_priority[self.tasks[0]] * tloss[self.tasks[0]],
                                     loss_priority[self.tasks[1]] * tloss[self.tasks[1]],
                                     loss_priority[self.tasks[2]] * tloss[self.tasks[2]] ])


        # -------------
        self.clear_masked_gradient(self.model.backbone)
        self.clear_masked_gradient(self.model.heads['segment_semantic'])
        self.clear_masked_gradient(self.model.heads['normal'])
        self.clear_masked_gradient(self.model.heads['depth_zbuffer'])
        # -------------

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return


    # ============================ finetune helper ==========================================
    # Not update the gradiant where the mask==0, by applying the mask to the gradiant
    def clear_masked_gradient(self, pruned_model):
        for k, m in enumerate(pruned_model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)

    
    def validate(self, it=0, writer=None, savePath=None, loss_priority=None):
        global best_acc
        self.model.eval()
        loss_list = {}
        for task in self.tasks:
            loss_list[task] = []
        
        for i, data in enumerate(self.val_dataloader):  # we have total 41 images in the validation dataloader
            x = data['input'].cuda()
            output = self.model(x)

            for task in self.tasks:
                y = data[task].cuda()       # y is the ground truth
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
            logging.info('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], avg_loss))
            logging.info(val_results)
        if self.early_stop:
            self.early_stop_monitor(task_val_results)
        logging.info('*' * 80)

        # save the best acc model
        test_acc = 0
        mIoU, Pixel_acc = task_val_results['segment_semantic']['mIoU'], task_val_results['segment_semantic']['Pixel Acc']
        angle_mean, angle_median, angle1, angle2, angle3 = task_val_results['normal']['Angle Mean'], task_val_results['normal']['Angle Median'], task_val_results['normal']['Angle 11.25'], task_val_results['normal']['Angle 22.5'], task_val_results['normal']['Angle 30']
        abs_err, rel_err, sigma1, sigma2, sigma3 = task_val_results['depth_zbuffer']['abs_err'], task_val_results['depth_zbuffer']['rel_err'], task_val_results['depth_zbuffer']['sigma_1.25'], task_val_results['depth_zbuffer']['sigma_1.25^2'], task_val_results['depth_zbuffer']['sigma_1.25^3']
        
        if self.architecture == 'resnet34' or 'resnet':
            seg1, seg2 = 0.2712, 0.5913
            norm1, norm2, norm3, norm_mean, norm_median = 29.40, 72.30, 87.30, 17.70, 16.30
            dep1, dep2, dep3, dep_abs, dep_rel = 57.91, 86.61, 96.37, 0.61, 0.23
        else:        # mobilenetv2
            seg1, seg2 = 0.2036, 0.4944
            norm1, norm2, norm3, norm_mean, norm_median = 28.37, 70.20, 85.58, 18.17, 16.62
            dep1, dep2, dep3, dep_abs, dep_rel = 47.92, 78.46, 92.81, 0.77, 0.28
        
        sum = loss_priority['segment_semantic'] + loss_priority['normal'] + loss_priority['depth_zbuffer']

        # the denominator of each metric is the independent model performance score
        test_acc = loss_priority['segment_semantic'] / sum * 0.5 * (mIoU/seg1 + Pixel_acc/seg2) + \
                    loss_priority['normal'] / sum * (angle1/norm1 + angle2/norm2 + angle3/norm3 - angle_mean/norm_mean - angle_median/norm_median) + \
                    loss_priority['depth_zbuffer'] / sum * (sigma1/dep1 + sigma2/dep2 + sigma3/dep3 - abs_err/dep_abs - rel_err/dep_rel)
        logging.info("test score: {}".format(round(test_acc, 4)))

        # record the best MTL accuracy
        if test_acc > best_acc:
            self.save_model(it, savePath + 'best_')
        best_acc = max(best_acc, test_acc)

        return
    
    def early_stop_monitor(self, task_val_results):
        rel_perm = {}
        better = 0
        for task in self.tasks:
            idx = 0
            temp = 0
            for metric in task_val_results[task]:
                idx += 1
                refer = self.metric_dict[task].refer[metric]
                prop = self.metric_dict[task].metric_prop[metric] #True: Lower the better
                value = task_val_results[task][metric]
                if prop:
                    if refer > value:
                        better += 1
                    temp += (refer - value)/refer*100
                else:
                    if refer < value:
                        better += 1
                    temp += (value - refer)/refer*100
            rel_perm[task] = temp/idx
        logging.info(rel_perm)
        
        overall = sum(rel_perm[key] for key in rel_perm)/len(rel_perm)
        if better >= self.good_metric and overall > 0.:
            self.counter += 1
        else:
            self.counter = 0
        return
    
    # helper functions: clear the training loss
    def set_train_loss(self):
        for task in self.tasks:
            self.loss_list[task] = []
        self.loss_list['total'] = []
        return
    
    def load_model(self, savePath, reload):
        model_name = True
        for task in self.tasks:
            if task not in reload:
                model_name = False
                break
        if model_name:
            # state = torch.load(savePath + reload)
            state = torch.load(reload)
            if type(state['iter']) is str:
                self.startIter = 0
            else:
                self.startIter = state['iter'] + 1
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            print('Cannot load from models trained from different tasks.')
            exit()
        return
    
    def save_model(self, it, savePath):
        state = {'iter': it,
                'state_dict': self.model.state_dict(),
                'layout': self.model.layout,
                'optimizer': self.optimizer.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
        if hasattr(self.model, 'branch') and self.model.branch is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '_b' + str(self.model.branch) + '.model')
        elif hasattr(self.model, 'layout') and self.model.layout is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '.model')
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
            logging.info('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss))
        logging.info('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), np.mean(self.loss_list['total'])))
        logging.info('======================================================================')
        return
