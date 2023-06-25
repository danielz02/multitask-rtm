#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import copy

import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

from models.aspp import MLPHead
from utils.custom_collate import collate_mil

"""
    Model getters 
"""


def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'resnet18':
        from models.resnet1d import resnet18
        backbone = resnet18()
        backbone_channels = 512

    elif p['backbone'] == 'resnet50':
        from models.resnet1d import resnet50
        backbone = resnet50()
        backbone_channels = 2048

    elif p['backbone'] == 'hrnet_w18':
        from models.seg_hrnet import hrnet_w18
        backbone = hrnet_w18()
        backbone_channels = [18, 36, 72, 144]

    else:
        raise NotImplementedError

    if p['backbone_kwargs']['dilated']:  # Add dilated convolutions
        assert (p['backbone'] in ['resnet18', 'resnet50'])
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    if 'fuse_hrnet' in p['backbone_kwargs'] and p['backbone_kwargs']['fuse_hrnet']:
        # Fuse the multi-scale HRNet features
        from models.seg_hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """

    if p['head'] == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])

    elif p['head'] == 'hrnet':
        from models.seg_hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    elif p['head'] == "MLP":
        return MLPHead(backbone_channels)
    else:
        raise NotImplementedError


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)

    if p['setup'] == 'single_task':
        from models.models import SingleTaskModel
        task = p.TASKS.NAMES[0]
        head = get_head(p, backbone_channels, task)
        model = SingleTaskModel(backbone, head, task)

    elif p['setup'] == 'multi_task':
        if p['model'] == 'baseline':
            from models.models import MultiTaskModel
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MultiTaskModel(backbone, heads, p.TASKS.NAMES)

        elif p['model'] == 'cross_stitch':
            from models.models import SingleTaskModel
            from models.cross_stitch import CrossStitchNetwork

            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in p.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(p, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(
                    os.path.join(p['root_dir'], p['train_db_name'], p['backbone'], 'single_task', task,
                                 'best_model.pth.tar')))
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder

            # Stitch the single-task models together
            model = CrossStitchNetwork(p, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict),
                                       **p['model_kwargs']['cross_stitch_kwargs'])

        elif p['model'] == 'nddr_cnn':
            from models.models import SingleTaskModel
            from models.nddr_cnn import NDDRCNN

            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in p.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(p, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(
                    os.path.join(p['root_dir'], p['train_db_name'], p['backbone'], 'single_task', task,
                                 'best_model.pth.tar')))
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder

            # Stitch the single-task models together
            model = NDDRCNN(p, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict),
                            **p['model_kwargs']['nddr_cnn_kwargs'])

        elif p['model'] == 'mtan':
            from models.mtan import MTAN
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTAN(p, backbone, heads, **p['model_kwargs']['mtan_kwargs'])

        elif p['model'] == 'pad_net':
            from models.padnet import PADNet
            model = PADNet(p, backbone, backbone_channels)

        elif p['model'] == 'mti_net':
            from models.mti_net import MTINet
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTINet(p, backbone, backbone_channels, heads)

        else:
            raise NotImplementedError('Unknown model {}'.format(p['model']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))

    return model


"""
    Transformations, datasets and dataloaders
"""


def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms as tr

    # Training transformations
    if p['train_db_name'] == 'NYUD':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    elif p['train_db_name'] == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    elif p['train_db_name'] == 'PROSPECT':
        transforms_tr = [torch.nn.Identity()]
    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))

    # Fixed Resize to input resolution
    # transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
    #                                      flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    # transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
    #                       tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transforms_tr = transforms.Compose(transforms_tr)

    # Testing (during training transforms)
    # transforms_ts = []
    # transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TEST.SCALE) for x in p.TASKS.FLAGVALS},
    #                                      flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    # transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
    #                       tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transforms_ts = transforms.Compose(transforms_ts)
    transforms_ts = [torch.nn.Identity()]

    return transforms_tr, transforms_ts


def get_train_dataset(p, transforms, ratio):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))
    if db_name == 'PROSPECT':
        from data.tabular import TabularRegression
        database = TabularRegression(
            path="./dataset/data.csv", task_names=p.ALL_TASKS.NAMES, split_ratio=ratio, split="train"
        )
    else:
        raise NotImplemented("train_db_name: Dataset not supported")

    return database


def get_train_dataset_name(p, transforms, ratio, name):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'PROSPECT':
        from data.tabular import TabularRegression
        database = TabularRegression(
            path="./dataset/data.csv", task_names=name, split_ratio=ratio, split="train"
        )
    else:
        raise NotImplemented("train_db_name: Dataset not supported")

    return database


def get_train_dataloader(p, dataset):
    """ Return the train dataloader """
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], shuffle=True, drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate_mil)
    return trainloader


def get_val_dataset(p, transforms):
    """ Return the validation dataset """

    db_name = p['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'PROSPECT':
        from data.tabular import TabularRegression
        database = TabularRegression(
            path="./dataset/data.csv", task_names=p.ALL_TASKS.NAMES, split_ratio=0.8, split="val"
        )
    else:
        raise NotImplemented("train_db_name: Dataset not supported")

    return database


def get_val_dataloader(p, dataset):
    """ Return the validation dataloader """

    testloader = DataLoader(
        dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False, num_workers=p['nworkers']
    )
    return testloader


""" 
    Loss functions 
"""


def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if p["train_db_name"] == "PROSPECT":
        from torch.nn import MSELoss
        criterion = MSELoss(reduction='none')
    else:
        raise NotImplementedError('Undefined Loss: Choose a task among PROSPECT')

    return criterion


def get_criterion(p):
    """ Return training criterion for a given setup """

    if p['setup'] == 'single_task':
        from losses.loss_schemes import SingleTaskLoss
        task = p.TASKS.NAMES[0]
        loss_ft = get_loss(p, task)
        return SingleTaskLoss(loss_ft, task)

    elif p['setup'] == 'multi_task':
        if p['loss_kwargs']['loss_scheme'] == 'baseline':  # Fixed weights
            from losses.loss_schemes import MultiTaskLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return MultiTaskLoss(p.TASKS.NAMES, loss_ft, loss_weights)

        elif p['loss_kwargs']['loss_scheme'] == 'pad_net':  # Fixed weights but w/ deep supervision
            from losses.loss_schemes import PADNetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.ALL_TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return PADNetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)

        elif p['loss_kwargs']['loss_scheme'] == 'mti_net':  # Fixed weights but at multiple scales
            from losses.loss_schemes import MTINetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in set(p.ALL_TASKS.NAMES)})
            loss_weights = p['loss_kwargs']['loss_weights']
            return MTINetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(p['loss_kwargs']['loss_scheme']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))


def get_criterion_single(task):
    """ Return training criterion for a given setup """

    from losses.loss_schemes import SingleTaskLoss
    from torch.nn import MSELoss
    criterion = MSELoss(reduction='none')
    return SingleTaskLoss(criterion, task)


"""
    Optimizers and schedulers
"""


def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    if p['model'] == 'cross_stitch':  # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert (p['optimizer'] == 'sgd')  # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100 * p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                    momentum=p['optimizer_kwargs']['momentum'],
                                    nesterov=p['optimizer_kwargs']['nesterov'],
                                    weight_decay=p['optimizer_kwargs']['weight_decay'])

    elif p['model'] == 'nddr_cnn':  # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert (p['optimizer'] == 'sgd')  # Adam seems to fail for nddr-cnns
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100 * p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                    momentum=p['optimizer_kwargs']['momentum'],
                                    nesterov=p['optimizer_kwargs']['nesterov'],
                                    weight_decay=p['optimizer_kwargs']['weight_decay'])

    else:  # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()

        if p['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

        elif p['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

        else:
            raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
