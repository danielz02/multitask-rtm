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
