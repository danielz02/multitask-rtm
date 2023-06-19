#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskLoss(nn.Module):
    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    def forward(self, pred, gt):
        cur_pred = pred[self.task].reshape(-1)
        cur_gt = gt[self.task].reshape(-1)
        valid = (cur_gt != -999).squeeze()
        cur_pred = cur_pred[valid]
        cur_gt = cur_gt[valid]
        out = {self.task: torch.nanmean(self.loss_ft(cur_pred, cur_gt))}
        out['total'] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {}
        for task in self.tasks:
            temp = gt[task].reshape(-1)
            valid = (temp != -999).squeeze()
            cur_pred = pred[task].reshape(-1)[valid]
            cur_gt = gt[task].reshape(-1)[valid]
            out[task] = torch.nanmean(self.loss_ft[task](cur_pred, cur_gt))
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        
#         out = {}
#         for task in self.tasks:
#             out[task] = torch.nanmean(self.loss_ft[task](pred[task].reshape(-1), gt[task].reshape(-1)))
#         out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        return out


class PADNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                 loss_weights: dict):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_ = F.interpolate(pred['initial_%s' % task], img_size, mode='bilinear')
            gt_ = gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out['deepsup_%s' % task] = loss_
            total += self.loss_weights[task] * loss_

        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out['total'] = total

        return out


class MTINetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                 loss_weights: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions at multiple scales (deepsup)
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' % (scale)]
            pred_scale = {t: F.interpolate(pred_scale[t], img_size, mode='bilinear') for t in self.auxilary_tasks}
            losses_scale = {t: self.loss_ft[t](pred_scale[t], gt[t]) for t in self.auxilary_tasks}
            for k, v in losses_scale.items():
                out['scale_%d_%s' % (scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v

        out['total'] = total

        return out
