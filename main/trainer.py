import logging
import numpy as np
from sys import exit
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models.Deeplab_ResNet34 import STRConv
from args import args
# add a better optimizer
from main.pcgrad import PCGrad
import bisect       # for list insertion

if args.dataset == "taskonomy":
    # the error is the lower the better
    best_acc = 10
    top10_acc = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
elif args.dataset == "NYUV2":
    # the accuracy is the higher the better
    best_acc = -10
    top10_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


class Trainer():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict,
                 architecture = 'resnet', logging=None,
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 val_iters=5000, print_iters=50, save_iters=2000,
                 early_stop=False, stop=3, good_metric=10,
                 adaptive_loss_begin=5000,
                #  adaptive_loss_begin=100,
                 sliding_window=500):
        super(Trainer, self).__init__()
        self.model = model
        self.startIter = 0
        # weight_decay=0.0002, original_weight_decay=0.0001
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
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
        self.set_train_loss()

        # Initialize the previous epoch loss as None
        self.prev_loss = {task: None for task in self.tasks}
        self.loss_difference = {task: [] for task in self.tasks}
        self.loss_weight = {task: 1 for task in self.tasks}
        self.adaptive_loss_begin = adaptive_loss_begin
        self.sliding_window = sliding_window
        
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
        global top10_acc

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
            # self.load_model(savePath, reload)
            self.load_dense_model(savePath, reload)

        # assign the priority for different tasks
        loss_priority = {}
        if priority is None:
            loss_priority = {task: 1 for task in self.tasks}
        else:
            for i in range(len(self.tasks)):
                loss_priority[self.tasks[i]] = priority[i]
        logging.info(loss_priority)

        # validate before training to avoid the out-of-memory
        self.validate(0, writer, savePath, loss_priority)

        for i in range(self.startIter, iters):
            if self.early_stop and self.counter >= self.stop:
                if savePath is not None:
                    self.save_model(i, savePath)
                logging.info('Early Stop Occur at {} Iter'.format((i+1)))
                break
            
            # self.train_step(loss_priority, i)
            self.model.train()
            try:
                data = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                data = next(self.train_iter)
                
            x = data['input'].cuda()
            # Forward pass
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
                
                # --------- Calculate the weights for the current epoch loss ---------
                if i >= self.adaptive_loss_begin:
                    if self.prev_loss[task] is not None:
                        # self.loss_difference[task].append(abs(self.prev_loss[task] - tloss[task].item()))     # abs(prev_loss - curr_loss)
                        # weight normalization
                        self.loss_difference[task].append(abs(self.prev_loss[task] - tloss[task].item()) / tloss[task].item())     # abs(prev_loss - curr_loss)

                    self.prev_loss[task] = tloss[task].item()
                    if len(self.loss_difference[task]) > self.sliding_window:
                        self.loss_difference[task].pop(0)
                        # get the average loss weights for different tasks
                        self.loss_weight[task] = sum(self.loss_difference[task]) / len(self.loss_difference[task])
                # ---------

            # normalize the loss_weight
            sum_weight = sum(self.loss_weight.values())
            for task in self.tasks:
                self.loss_weight[task] = self.loss_weight[task] * 3 / sum_weight
                writer.add_scalar('LR/' + task + '/weights', self.loss_weight[task], i)


            # update the total loss using adaptive weight09
            for task in self.tasks:
                loss += loss_priority[task] * tloss[task] * self.loss_weight[task]
            
            self.loss_list['total'].append(loss.item())
            
            loss.backward()

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

            # ========== store the sparsity for different components =========
            s1, t1 = self.store_sparsity(self.model.backbone, writer, i, 'Backbone')
            s2, t2 = self.store_sparsity(self.model.heads['segment_semantic'], writer, i, 'Seg')
            s3, t3 = self.store_sparsity(self.model.heads['normal'], writer, i, 'Nor')
            s4, t4 = self.store_sparsity(self.model.heads['depth_zbuffer'], writer, i, 'Dep')

            overall_sparsity = (s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4) / 100.0 / (t1 + t2 + t3 + t4)
            writer.add_scalar("sparsity/overall_sparsity", overall_sparsity, i)
            writer.add_scalar("LR/sparseThreshold", self.model.backbone.sparseThreshold, i)



            # stop updating the sparseThreshold if the current sparsity has reached the desired sparsity
            # if i > iters * 0.8:
            if overall_sparsity >= args.ratio:
                    self.model.backbone.sparseThreshold.grad = None
                    self.model.heads['segment_semantic'].sparseThreshold.grad = None
                    self.model.heads['normal'].sparseThreshold.grad = None
                    self.model.heads['depth_zbuffer'].sparseThreshold.grad = None


            # Update weights
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Clear gradients
            self.optimizer.zero_grad()

        # Reset loss list and the data iters
        self.set_train_loss()
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


    # Storing sparsity and threshold statistics for STRConv models
    def store_sparsity(self, check_model, writer, iter, name):

        # Storing sparsity and threshold statistics for STRConv models
        if args.conv_type == "STRConv":
            count = 0
            sum_sparse = 0.0
            for n, m in check_model.named_modules():
                if isinstance(m, STRConv):
                    sparsity, total_params, thresh = m.getSparsity()
                    writer.add_scalar("Layer_sparsity/{}_{}".format(name, n), sparsity, iter)
                    writer.add_scalar("Layer_thresh/{}_{}".format(name, n), thresh, iter)
                    sum_sparse += int(((100 - sparsity) / 100) * total_params)
                    count += total_params
            total_sparsity = 100 - (100 * sum_sparse / count)
            writer.add_scalar("sparsity/{}".format(name), total_sparsity, iter)

            return total_sparsity, count
        # ======================================================================


    
    def validate(self, it=0, writer=None, savePath=None, loss_priority=None):
        global best_acc
        global top10_acc
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
        
        # write tensorboard: learning rate
        cur_lr = self.optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", cur_lr, it)
        writer.add_scalar("LR/weight_decay", args.weight_decay, it)

        task_val_results = {}
        for task in self.tasks:
            avg_loss = np.mean(loss_list[task])
            val_results = self.metric_dict[task].val_metrics()
            if writer != None:
                writer.add_scalar('Loss/val/' + task, avg_loss, it)
                for metric in val_results:
                    writer.add_scalar('Metric/' + task + '/' + metric, val_results[metric], it)
            # if self.early_stop:

            # logging write
            task_val_results[task] = val_results
            logging.info('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], avg_loss))
            logging.info(val_results)

        if self.early_stop:
            self.early_stop_monitor(task_val_results)
        logging.info('*' * 80)

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
            
            
            # sort the top10_acc
            bisect.insort(top10_acc, round(test_error, 4))
            top10_acc.pop(-1)

            logging.info("test error score: {}".format(round(test_error, 4)))
            logging.info("top 10 test error score: {}".format(top10_acc))
            writer.add_scalar('Loss/average_error', round(test_error, 4), it)

            # we only save the best models after 16,000 iters
            if it > 10000:
                # record the best MTL accuracy. The error is the lower, the better
                if test_error < best_acc:
                    self.save_model(it, savePath + 'best_')
                best_acc = min(best_acc, test_error)
            
            
            
            logging.info("average_error: {}".format(round(test_error, 4)))


        elif args.dataset == "NYUV2":
            # save the best acc model
            test_acc = 0
            mIoU, Pixel_acc = task_val_results['segment_semantic']['mIoU'], task_val_results['segment_semantic']['Pixel Acc']
            angle_mean, angle_median, angle1, angle2, angle3 = task_val_results['normal']['Angle Mean'], task_val_results['normal']['Angle Median'], task_val_results['normal']['Angle 11.25'], task_val_results['normal']['Angle 22.5'], task_val_results['normal']['Angle 30']
            abs_err, rel_err, sigma1, sigma2, sigma3 = task_val_results['depth_zbuffer']['abs_err'], task_val_results['depth_zbuffer']['rel_err'], task_val_results['depth_zbuffer']['sigma_1.25'], task_val_results['depth_zbuffer']['sigma_1.25^2'], task_val_results['depth_zbuffer']['sigma_1.25^3']
            
            if self.architecture == 'resnet34':
                seg1, seg2 = 0.2525, 0.5773
                norm_mean, norm_median, norm1, norm2, norm3  = 17.2398, 14.9797, 36.433, 72.0786, 85.2728
                dep_abs, dep_rel, dep1, dep2, dep3 = 0.5551, 0.2151, 64.4659, 89.8674, 97.4203
            else:        # mobilenetv2
                seg1, seg2 = 0.1849, 0.4864
                norm_mean, norm_median, norm1, norm2, norm3  = 17.7522, 16.2718, 28.5603, 73.9923, 87.3249
                dep_abs, dep_rel, dep1, dep2, dep3 = 0.587, 0.2404, 61.9075, 87.4701, 96.1797
            
            sum = loss_priority['segment_semantic'] + loss_priority['normal'] + loss_priority['depth_zbuffer']

            # the denominator of each metric is the independent model performance score
            # [8 metrics] the denominator of each metric is the independent model performance score
            # test_acc = loss_priority['segment_semantic'] / sum * 0.5 * (mIoU/seg1 + Pixel_acc/seg2) + \
            #             loss_priority['normal'] / sum * 0.33 * (angle1/norm1 + angle2/norm2 + angle3/norm3) + \
            #             loss_priority['depth_zbuffer'] / sum * 0.33 * (sigma1/dep1 + sigma2/dep2 + sigma3/dep3)
            # 12 metrics
            test_acc = loss_priority['segment_semantic'] / sum * 0.5 * (mIoU/seg1 + Pixel_acc/seg2) + \
                        loss_priority['normal'] / sum * (angle1/norm1 + angle2/norm2 + angle3/norm3 - angle_mean/norm_mean - angle_median/norm_median) + \
                        loss_priority['depth_zbuffer'] / sum * (sigma1/dep1 + sigma2/dep2 + sigma3/dep3 - abs_err/dep_abs - rel_err/dep_rel)
            
            # sort the top10_acc
            bisect.insort(top10_acc, round(test_acc, 4))
            top10_acc.pop(0)

            logging.info("test score: {}".format(round(test_acc, 4)))
            logging.info("top 10 test score: {}".format(top10_acc))
            writer.add_scalar('Loss/avg_score', round(test_acc, 4), it)

            # we only save the best models after 16,000 iters
            if it > 16000:
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
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            print('Cannot load from models trained from different tasks.')
            exit()
        return


    def load_dense_model(self, savePath, reload):
        model_name = True
        for task in self.tasks:
            if task not in reload:
                model_name = False
                break
        if model_name:
            # state = torch.load(savePath + reload)
            state = torch.load(reload)
            self.model.load_state_dict(state['state_dict'])
            
            # xmc_modify here for different dataset
            # assign new values
            self.model.backbone.sparseThreshold.data = torch.tensor(args.sInit_value).cuda()
            self.model.heads['segment_semantic'].sparseThreshold.data = torch.tensor(args.sInit_value).cuda()
            self.model.heads['normal'].sparseThreshold.data = torch.tensor(args.sInit_value).cuda()
            self.model.heads['depth_zbuffer'].sparseThreshold.data = torch.tensor(args.sInit_value).cuda()

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
