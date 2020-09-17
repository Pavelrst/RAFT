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

    result_dict = {'epe_clean': -1, 'epe_final': -1}

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
        if dstype == 'clean':
            result_dict['epe_clean'] = np.mean(epe_list)
        elif dstype == 'final':
            result_dict['epe_final'] = np.mean(epe_list)
        else:
            raise NotImplementedError

    return  result_dict


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
    '''
    This function validates all skpts in all subdirectories of CKPT_PATH
    '''
    PATH = '../RAFT_golden_ckpts'

    raft_ckpts = []
    for subdir in os.listdir(PATH):
        subdir_path = os.path.join(PATH, subdir)
        for ckpt_name in os.listdir(subdir_path):
            if '.pth' in ckpt_name:
                raft_ckpts.append(os.path.join(subdir_path, ckpt_name))

    print('Validate on following ckpt:')
    for ckpt_name in raft_ckpts:
        print(ckpt_name)

    res_dict = dict(zip(raft_ckpts, np.zeros(len(raft_ckpts))))

    # Validate RAFT augmentor per steps
    for ckpt in res_dict.keys():
        print('============ validate', ckpt)
        args = raft_sintel_val_args(ckpt)

        model = RAFT(args)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model))

        model.to('cuda')
        model.eval()

        res_dict[ckpt] = validate_sintel(args, model, args.iters)

    # build plot
    labels = [os.path.basename(x) for x in res_dict.keys()]
    epe_clean = [np.round(res_dict[key]['epe_clean'], 2) for key in res_dict.keys()]
    epe_final = [np.round(res_dict[key]['epe_final'], 2) for key in res_dict.keys()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, epe_clean, width, label='epe_clean')
    rects2 = ax.bar(x + width / 2, epe_final, width, label='epe_final')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('EPE')
    ax.set_title('EPE comparison, -b2 -iter240K')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.tight_layout()
    plt.savefig('foo.png')
