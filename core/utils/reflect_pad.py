import numpy as np

def reflect_pad(img):
    '''
    :param img: image of shape [H, W, C]
    :return: padded image of shape [H, W, C]
    '''
    ch_num = img.shape[2]

    TOP_PAD = 0
    DOWN_PAD = int(img.shape[0]*0.25)
    LEFT_PAD = 0 #int(img.shape[1]*0.125)
    RIGHT_PAD = 0 #int(img.shape[1]*0.125)

    res = np.zeros((img.shape[0]+TOP_PAD+DOWN_PAD, img.shape[1]+LEFT_PAD+RIGHT_PAD, img.shape[2]))

    for ch in range(ch_num):
        res[:, :, ch] = np.pad(img[:, :, ch], ((TOP_PAD, DOWN_PAD), (LEFT_PAD, RIGHT_PAD)), 'reflect')

    return res.astype(np.uint8)

def refect_pad_flow(flow):
    '''
    :param flow: image of shape [H, W, C]
    :return: padded image of shape [H, W, C]
    '''
    TOP_PAD = 0
    DOWN_PAD = int(flow.shape[0]*0.25)
    LEFT_PAD = 0 #int(flow.shape[1]*0.125)
    RIGHT_PAD = 0 #int(flow.shape[1]*0.125)

    res = flow
    bottom_pad = np.flip(flow, axis=0)
    bottom_pad = bottom_pad[range(DOWN_PAD), :, :]
    # negate channel 1
    bottom_pad[:, :, 1] = -bottom_pad[:, :, 1]
    res = np.concatenate([res, bottom_pad], axis=0)

    return res



