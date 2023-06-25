#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch
from pcgrad import PCGrad
import pandas as pd
import torch.backends.cudnn
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations, \
    get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_optimizer, get_model, adjust_learning_rate, \
    get_criterion
from utils.logger import Logger
from train.train_utils import train_vanilla, cosine_similarity
from evaluation.evaluate_utils import eval_model
from termcolor import colored



def main():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))

    # Get model
    model = get_model(p)
    model = torch.nn.DataParallel(model)

    # Get criterion
    criterion = get_criterion(p)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer = get_optimizer(p, model)

    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms, 0.8)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None)  # True validation dataset without reshape
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0

    # Main loop
    print(colored('Starting main loop', 'blue'))

    best_result = -99999
    optimizer = PCGrad(optimizer)

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer.optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        train_vanilla(p, train_dataloader, model, criterion, optimizer, epoch)
        eval_results = eval_model(p, val_dataloader, model)
        print(best_result)
        print(np.mean([x["r2"] for x in eval_results.values()]))
        if np.mean([x["r2"] for x in eval_results.values()]) > best_result:
            best_result = np.mean([x["r2"] for x in eval_results.values()])
            torch.save({'optimizer': optimizer.optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1, 'best_result': best_result}, p['best_model'])

        # Checkpoint
        print('Checkpoint ...')
        torch.save(
            {'optimizer': optimizer.optimizer.state_dict(), 'total_optimizer': optimizer, 'model': model.state_dict(),
             'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])


def get_target_pred():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    # model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    # criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms 
    val_dataloader = torch.load('val_loader.pt')

    # Resume from checkpoint
    checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']

    # Main loop
    print(colored('Starting main loop', 'blue'))
    tasks = p.TASKS.NAMES
    pred = {}
    target = {}
    temp = len(val_dataloader)
    for i, batch in enumerate(val_dataloader):
        print(str(i) + '/' + str(temp))
        # Forward pass
        images = batch['image']  # .cuda(non_blocking=True)
        targets = {task: batch[task] for task in tasks}  # .cuda(non_blocking=True)
        output = model(images)
        for t in tasks:
            temp1 = pred.get(t, 'empty')
            temp2 = target.get(t, 'empty')
            if temp1 != 'empty':
                pred[t] = np.append(pred[t], output[t].flatten().cpu().data.numpy())
                target[t] = np.append(target[t], targets[t].flatten().cpu().data.numpy())
            else:
                pred[t] = output[t].flatten().cpu().data.numpy()
                target[t] = targets[t].flatten().cpu().data.numpy()

    torch.save(target, 'target.pth')
    torch.save(pred, 'pred.pth')


def get_cosine_similarity():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))

    # Get model
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    print('finish get model')

    # Get criterion
    criterion = get_criterion(p)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer = get_optimizer(p, model)
    print('optimizer')

    # Transforms
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms, 1)
    # from torch.utils.data import DataLoader
    # from utils.custom_collate import collate_mil
    # train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset)/5, shuffle=True, drop_last=True,
    #                          num_workers=p['nworkers'], collate_fn=collate_mil)
    train_dataloader = get_train_dataloader(p, train_dataset)

    # Resume from checkpoint
    checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    print('resume from checkpoint')
    optimizer = PCGrad(optimizer)

    grad, task = cosine_similarity(p, train_dataloader, model, criterion, optimizer, 0)
    result = pd.DataFrame(grad, columns=task)
    result.to_csv('cosine_similarity_result.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vanilla Training')
    parser.add_argument('--config-env',
                        help='Config file for the environment')
    parser.add_argument('--config-exp',
                        help='Config file for the experiment')
    args = parser.parse_args()

    # main()
    get_target_pred()
    # test()
