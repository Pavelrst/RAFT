import numpy as np

def reflect_pad(img, flow=False):
    '''
    :param img: image of shape [H, W, C]
    :return: padded image of shape [H, W, C]
    '''
    ch_num = img.shape[2]

    TOP_PAD = 0
    DOWN_PAD = int(img.shape[0]*0.25)
    LEFT_PAD = int(img.shape[1]*0.125)
    RIGHT_PAD = int(img.shape[1]*0.125)

    res = np.zeros((img.shape[0]+TOP_PAD+DOWN_PAD, img.shape[1]+LEFT_PAD+RIGHT_PAD, img.shape[2]))

    for ch in range(ch_num):
        res[:, :, ch] = np.pad(img[:, :, ch], ((TOP_PAD, DOWN_PAD), (LEFT_PAD, RIGHT_PAD)), 'reflect')

    if flow == True:
        return res
    else:
        return res.astype(np.uint8)





