import numpy as np
import random
import math
import cv2
from PIL import Image

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from core.utils.scopeflow_utils.scopeflow_augmentor import RandomAffineFlowOccSintel
from core.utils.scopeflow_utils import transforms
from torchvision import transforms as vision_transforms
from core.utils.scopeflow_utils.vis_utils import show_image
from core.utils.reflect_pad import reflect_pad
from core.utils.flow_viz import flow_to_image
from tqdm import tqdm


class ScopeFlowAugmentor_adapter:
    '''
    This class adapts the implementation
    of the augmentation technique
    described in ScopeFlow paper.
    '''

    def __init__(self, args):
        self.augmentor_module = RandomAffineFlowOccSintel(args, addnoise=True)
        self.show_aug = args.show_aug
        self.image_crop_size = args.image_size

        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------
        if args.photometric_augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> PIL
                vision_transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                vision_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=True, to_numpy=False)

        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> FloatTensor
                vision_transforms.transforms.ToTensor(),
            ], from_numpy=True, to_numpy=False)

    def adapt_input_to_scopeflow_augmentor(self, img1, img2, flow):
        '''
        Batchify and transpose to shape BxCHxHxW, convert to range [0,1]
        :param img: numpy array BxHxWxCH
        :return: numpy array BxCHxHxW
        '''
        if self.show_aug:
            plt.subplot(221)
            plt.title('1 (Input)')
            show_image(img1, subplote=True)

        img1, img2 = self._photometric_transform(img1, img2)
        flow = np.transpose(flow, (2, 0, 1))
        flow = np.expand_dims(flow, 0)

        input_dict = {}
        input_dict['input1'] = img1.unsqueeze(0)
        input_dict['input2'] = img2.unsqueeze(0)
        input_dict['target1'] = torch.from_numpy(flow)
        input_dict['target_occ1'] = torch.from_numpy(flow)  # not used.

        return input_dict

    def readapt_input_to_raft(self, result_dict):
        '''
        Remove batch dimension, transpose to HxWxCH, set range [0, 255]
        '''
        img1 = result_dict['input1'].squeeze(0).cpu().detach().numpy()
        img2 = result_dict['input2'].squeeze(0).cpu().detach().numpy()
        flow = result_dict['target1'].squeeze(0).cpu().detach().numpy()

        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        flow = np.transpose(flow, (1, 2, 0))

        img1 = img1 * 255
        img2 = img2 * 255

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        input_dict = self.adapt_input_to_scopeflow_augmentor(img1, img2, flow)

        result_dict = self.augmentor_module.forward(input_dict)

        # Check that image size is of correct size, must be multiples of 8.
        while not result_dict['input1'].shape[2] % 8 == 0 and result_dict['input1'].shape[3] % 8 == 0:
            print("Fixing mul8 loop activated")
            result_dict = self.augmentor_module.forward(input_dict)

        return self.readapt_input_to_raft(result_dict)


class FlowAugmentor:
    '''
    This is an original augmentor used in RAFT model.
    '''

    def __init__(self, args):
        self.crop_size = args.image_size
        assert not isinstance(self.crop_size, str), 'args.image_size should not be string for RAFT augmentor'
        self.augcolor = torchvision.transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.5 / 3.14)

        self.asymmetric_color_aug_prob = 0.2
        self.spatial_aug_prob = 0.8
        self.eraser_aug_prob = 0.5

        self.min_scale = args.min_scale
        self.max_scale = args.max_scale
        self.max_stretch = 0.2
        self.stretch_prob = 0.8
        self.margin = 20

        self.show_aug = args.show_aug
        self.padding = args.padding

    def color_transform(self, img1, img2):

        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.augcolor(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.augcolor(Image.fromarray(img2)), dtype=np.uint8)

        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.augcolor(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        max_scale = self.max_scale
        min_scale = max(min_scale, self.min_scale)

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if np.random.rand() < 0.5:  # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]

        if np.random.rand() < 0.1:  # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(-self.margin, img1.shape[0] - self.crop_size[0] + self.margin)
        x0 = np.random.randint(-self.margin, img1.shape[1] - self.crop_size[1] + self.margin)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def reflection_padding(self, img1, img2, flow):
        '''
        :param input images: [H,W,Ch]
        :return: [H,W,Ch]
        '''
        img1 = reflect_pad(img1)
        img2 = reflect_pad(img2)
        flow = reflect_pad(flow, flow=True)
        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        if self.padding:
            img1, img2, flow = self.reflection_padding(img1, img2, flow)

        if self.show_aug:
            plt.subplot(221)
            plt.title('1 (Input)')
            show_image(img1, subplote=True)

        if self.show_aug:
            plt.subplot(222)
            plt.title('flow (Input)')
            show_image(flow_to_image(flow), subplote=True)

        img1, img2 = self.color_transform(img1, img2)

        img1, img2 = self.eraser_transform(img1, img2)

        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        if self.show_aug:
            plt.subplot(223)
            plt.title('1 (spatial)')
            show_image(img1, subplote=True)

        if self.show_aug:
            plt.subplot(224)
            plt.title('flow (spatial)')
            show_image(flow_to_image(flow))

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class FlowAugmentorKITTI:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5):
        self.crop_size = crop_size
        self.augcolor = torchvision.transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)

        self.max_scale = max_scale
        self.min_scale = min_scale

        self.spatial_aug_prob = 0.8
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.augcolor(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if np.random.rand() < 0.5:  # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
            valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
