import sys

sys.path.append('core')

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datasets
from raft import RAFT
from args import raft_sintel_val_args

def write_flow(filename, uv, v=None):
    '''
    :param filename: full path to the target file
    :param uv: optical flow numpy vector of shape [H, W, Ch]
    :param v: I don't know
    '''
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(np.array([202021.25], np.float32))
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def predict_sintel(args, model, test_dataset_path, iters=50):
    """ Predict optical flow for test set """
    model.eval()
    pad = 2

    flow_pred_dict = {'clean': [], 'final': []}

    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintelTest(args, root=test_dataset_path, dstype=dstype)
        assert len(test_dataset) > 0

        for i in tqdm(range(len(test_dataset))):
            image1, image2, sequence, frame = test_dataset[i]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            image1 = F.pad(image1, [0, 0, pad, pad], mode='replicate')
            image2 = F.pad(image2, [0, 0, pad, pad], mode='replicate')

            with torch.no_grad():
                flow_predictions = model.module(image1, image2, iters=iters)
                flow_pr = flow_predictions[-1][0, :, pad:-pad]

            flow_pred_dict[dstype].append({'flow_pred': flow_pr, 'sequence': sequence, 'frame': frame})

    return flow_pred_dict



if __name__ == '__main__':
    '''
    This function evaluates test dataset on 
    all skpts in all subdirectories of CKPT_PATH
    
    The directoty structure is like this:
    
    clean/ambush_1/frame0001.flo
              /frame0002.flo
              ...
     /ambush_3/frame0001.flo
              /frame0002.flo
              ...
     ...
     
    final/ambush_1/frame0001.flo
              /frame0002.flo
              ...
     /ambush_3/frame0001.flo
              /frame0002.flo
              ...
     ...
    '''

    PATH = '/home/pavelr/workspace/RAFT_golden_ckpts/' #'C:\\Users\\Pavel\\Desktop\\RAFT_golden_ckpts'
    TEST_DATASET = '/home/pavelr/workspace/RAFT/datasets/Sintel/test/' #'C:\\Users\\Pavel\\Downloads\\Sintel\\test'

    raft_ckpts = []
    for subdir in os.listdir(PATH):
        subdir_path = os.path.join(PATH, subdir)
        for ckpt_name in os.listdir(subdir_path):
            if '.pth' in ckpt_name:
                raft_ckpts.append(os.path.join(subdir_path, ckpt_name))

    res_dict = dict(zip(raft_ckpts, np.zeros(len(raft_ckpts))))

    # predict flows
    for ckpt in res_dict.keys():
        args = raft_sintel_val_args(ckpt)
        print('Predict', ckpt, ' refinement iterations: ', args.iters)

        model = RAFT(args)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model))

        model.to('cuda')
        model.eval()

        res_dict[ckpt] = predict_sintel(args, model, TEST_DATASET, args.iters)

    # save flow in files
    for ckpt in res_dict.keys():
        test_path = os.path.join(os.path.dirname(ckpt), 'test_predictions')
        try:
            os.mkdir(test_path)
        except:
            print(test_path, ' exists...')

        for subset_key in res_dict[ckpt].keys():
            print('Saving ', subset_key, ' subset')
            subset_path = os.path.join(test_path, subset_key)
            try:
                os.mkdir(subset_path)
            except:
                print(subset_path, ' exists...')

            # save flow predictions of the subset
            for flow_dict in res_dict[ckpt][subset_key]:
                flow_pred_tensor = flow_dict['flow_pred']
                sequence = flow_dict['sequence']
                frame = flow_dict['frame']

                sequence_path = os.path.join(subset_path, sequence)
                if not os.path.exists(sequence_path):
                    os.mkdir(sequence_path)

                filename = os.path.join(sequence_path, 'frame'+str(frame+1).zfill(4)+'.flo')
                print('Saving ', filename)

                flow_pred_nupmy = flow_pred_tensor.detach().cpu().numpy()
                flow_pred_nupmy = np.transpose(flow_pred_nupmy, axes=(1, 2, 0))
                write_flow(filename, flow_pred_nupmy)

