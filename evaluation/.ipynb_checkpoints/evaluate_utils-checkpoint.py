#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json

import pandas as pd
import torch
import scipy.io as sio

from evaluation.eval_regression import RegressionMeter
from utils.utils import get_output, mkdir_if_missing


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """

    def __init__(self, p):
        self.database = p['train_db_name']
        self.tasks = p.TASKS.NAMES
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            temp = gt[t].reshape(-1)
            valid = (temp != -999).squeeze()
            cur_pred = pred[t].reshape(-1)[valid]
            cur_gt = gt[t].reshape(-1)[valid]
            if len(cur_gt != 0):
                self.meters[t].update(cur_pred, cur_gt)

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert (set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse']) / stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']:  # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU']) / stl['mIoU']

        elif task == 'normals':  # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean']) / stl['mean']

        elif task == 'edge':  # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF']) / stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks


def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter()

    elif task == 'edge':  # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'])

    elif p["train_db_name"] == "PROSPECT":
        return RegressionMeter(target_name=task)
    else:
        raise NotImplementedError


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict.
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.
    """
    tasks = p.TASKS.NAMES

    if len(tasks) == 1:  # Single-task performance
        task = tasks[0]
        if task == 'semseg':  # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' % (
                    100 * reference['semseg']['mIoU'], 100 * current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' % (
                    100 * reference['semseg']['mIoU'], 100 * current['semseg']['mIoU']))
                improvement = False

        elif task == 'human_parts':  # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' % (
                    100 * reference['human_parts']['mIoU'], 100 * current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' % (
                    100 * reference['human_parts']['mIoU'], 100 * current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal':  # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' % (
                    100 * reference['sal']['mIoU'], 100 * current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' % (
                    100 * reference['sal']['mIoU'], 100 * current['sal']['mIoU']))
                improvement = False

        elif task == 'depth':  # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' % (
                    reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' % (
                    reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False

        elif task == 'normals':  # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' % (
                    reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' % (
                    reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge':  # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print(
                    'New best edge detection model %.3f -> %.3f' % (reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True

            else:
                print('No new best edge detection model %.3f -> %.3f' % (
                    reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False

    else:  # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' % (
                100 * reference['multi_task_performance'], 100 * current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' % (
                100 * reference['multi_task_performance'], 100 * current['multi_task_performance']))
            improvement = False

    if improvement:  # Return result
        return True, current

    else:
        return False, reference


@torch.no_grad()
def eval_model(p, val_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p)

    model.eval()

    for i, batch in enumerate(val_loader):
        # Forward pass
        images = batch['image']#.cuda(non_blocking=True)
        targets = {task: batch[task] for task in tasks} #.cuda(non_blocking=True)
        output = model(images)

        # Measure performance
        performance_meter.update({t: output[t] for t in tasks}, targets)

    eval_results = performance_meter.get_score(verbose=True)
    return eval_results


@torch.no_grad()
def save_model_predictions(p, val_loader, model):
    """ Save model predictions for all tasks """

    print('Save model predictions to {}'.format(p['save_dir']))
    model.eval()
    tasks = p.TASKS.NAMES
    task_output = {x: {"y_pred": [], "y_true": []} for x in tasks}
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)

    for ii, sample in enumerate(val_loader):
        inputs = sample['image']#.cuda(non_blocking=True)
        outputs = model(inputs)
        for t in tasks:
            task_output[t]["y_pred"].append(outputs[t].cpu().numpy())
            task_output[t]["y_true"].append(sample[t].cpu().numpy())

    for t in tasks:
        task_output[t]["y_pred"] = np.concatenate(task_output[t]["y_pred"]).reshape(-1)
        task_output[t]["y_true"] = np.concatenate(task_output[t]["y_true"]).reshape(-1)

        pd.DataFrame(task_output[t]).to_csv(os.path.join(save_dirs[t] + 'eval.csv'))

    return task_output


def eval_all_results(p, single_task_test_dict):
    """ Evaluate results for every task by reading the predictions from the save dir """

    results = {}
    if p['setup'] == 'multi_task':  # Perform multitask performance evaluation
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)
        print('Multi-task learning performance on test set is %.2f' % (100 * results['multi_task_performance']))

    return results
