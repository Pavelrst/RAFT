from __future__ import print_function, division
import sys
sys.path.append('core')
import numpy as np

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import neptune

from torch.utils.data import DataLoader
from raft import RAFT
from evaluate import validate_sintel, validate_kitti
import datasets
from args import raft_sintel_ft_args, raft_sintel_debug_args
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_loss(flow_preds, flow_gt, valid, args):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    valid = (valid >= 0.5) & (flow_gt.abs().sum(dim=1) < args.max_flow)

    for i in range(n_predictions):
        i_weight = 0.8**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'chairs':
        train_dataset = datasets.FlyingChairs(args, root=args.dataset_root, image_size=args.image_size)
    
    elif args.dataset == 'things':
        clean_dataset = datasets.SceneFlow(args, root=args.dataset_root, image_size=args.image_size, dstype='frames_cleanpass')
        final_dataset = datasets.SceneFlow(args, root=args.dataset_root, image_size=args.image_size, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        clean_dataset = datasets.MpiSintel_Train(args, root=args.dataset_root, image_size=args.image_size, dstype='clean')
        final_dataset = datasets.MpiSintel_Train(args, root=args.dataset_root, image_size=args.image_size, dstype='final')
        assert len(clean_dataset) == 908 and len(final_dataset) == 908
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'kitti':
        train_dataset = datasets.KITTI(args, root=args.dataset_root, image_size=args.image_size, is_val=False)

    else:
        raise NotImplementedError

    gpuargs = {'num_workers': args.num_of_workers, 'drop_last' : True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, **gpuargs)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps,
        pct_start=0.2, cycle_momentum=False, anneal_strategy='linear', final_div_factor=1.0)

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, sum_freq):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.sum_freq = sum_freq

    def _print_training_status(self):
        metrics_str = ''
        for key in sorted(self.running_loss.keys()):
            metric_val = (self.running_loss[key]/self.sum_freq)
            metrics_str += key+':'+str(np.round(metric_val, 2))+'  '
            # log to neptune
            neptune.log_metric(key, metric_val)
        training_str = "[total steps:{:6d}, lr:{:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        neptune.log_metric('learning rate', self.scheduler.get_lr()[0])

        # print the training status
        print(training_str + metrics_str)




        # zero running loss
        for key in self.running_loss:
            self.running_loss[key] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}


def train(args):
    model = RAFT(args)
    model = nn.DataParallel(model)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt))

    model.cuda()
    model.train()
    
    if 'chairs' not in args.dataset:
        model.module.freeze_bn()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler, args.sum_freq)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            optimizer.zero_grad()
            flow_predictions = model(image1, image2, iters=args.iters)
            
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps += 1

            logger.push(metrics)

            if total_steps % args.val_freq == args.val_freq-1:
                path = 'checkpoints/%d_%s.pth' % (total_steps+1, args.experiment_name)
                torch.save(model.state_dict(), path)

            if total_steps == args.num_steps:
                should_keep_training = False
                break

    path = 'checkpoints/%s.pth' % args.experiment_name
    torch.save(model.state_dict(), path)

    return path


if __name__ == '__main__':
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    neptune.init('pavelrasto/RAFT-Scopeflow')
    neptune.create_experiment(name='Scopeflow augmentor 1', description='Scopeflow augmentorr without photometric transformations -b3 -iter160K')

    args = raft_sintel_ft_args(exp_name='sintel_ft_sflo_aug_no_photometric', augment_type='scopeflow_augmentor')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # scale learning rate and batch size by number of GPUs
    num_gpus = torch.cuda.device_count()
    args.batch_size = args.batch_size * num_gpus
    args.lr = args.lr * num_gpus
    train(args)