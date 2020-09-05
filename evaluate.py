import sys
sys.path.append('core')

from PIL import Image
import cv2
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import datasets
from utils import flow_viz
from raft import RAFT

from args import raft_sintel_val_args

def validate_sintel(args, model, iters=50):
    """ Evaluate trained model on Sintel(train) clean + final passes """
    model.eval()
    pad = 2

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel_Val(args, root=args.dataset_root, do_augument=False, dstype=dstype)
        assert len(val_dataset) > 0

        epe_list = []
        for i in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[i]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            image1 = F.pad(image1, [0, 0, pad, pad], mode='replicate')
            image2 = F.pad(image2, [0, 0, pad, pad], mode='replicate')

            with torch.no_grad():
                flow_predictions = model.module(image1, image2, iters=iters)
                flow_pr = flow_predictions[-1][0,:,pad:-pad]

            epe = torch.sum((flow_pr - flow_gt.cuda())**2, dim=0)
            epe = torch.sqrt(epe).mean()
            epe_list.append(epe.item())

        print("Validation (%s) EPE: %f" % (dstype, np.mean(epe_list)))


def validate_kitti(args, model, iters=32):
    """ Evaluate trained model on KITTI (train) """

    model.eval()
    val_dataset = datasets.KITTI(args, do_augument=False, is_val=True, do_pad=True)

    with torch.no_grad():
        epe_list, out_list = [], []
        for i in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[i]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            flow_gt = flow_gt.cuda()
            valid_gt = valid_gt.cuda()

            flow_predictions = model.module(image1, image2, iters=iters)
            flow_pr = flow_predictions[-1][0]

            epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5

            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)


    print("Validation KITTI: %f, %f" % (np.mean(epe_list), 100*np.mean(out_list)))


if __name__ == '__main__':
    CKPT_PATH = '.\\checkpoints'
    raft_ckpts = []
    sflo_ckpts = []
    for ckpt_path in os.listdir(CKPT_PATH):
        if 'random_crop_v3' in ckpt_path:
            raft_ckpts.append(os.path.join(CKPT_PATH, ckpt_path))

    # Validate RAFT augmentor per steps
    for ckpt in raft_ckpts:
        # args = raft_sintel_val_args('.\\checkpoints\\sintel_ft_raft_aug.pth')
        print('============ validate', ckpt)
        args = raft_sintel_val_args(ckpt)

        model = RAFT(args)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model))

        model.to('cuda')
        model.eval()

        validate_sintel(args, model, args.iters)
