# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random
import torchvision.transforms.functional_tensor as TF

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop

from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "patches"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def random_image_box_translation(img, boxes):
    boxes = boxes.clone()
    im_shape = img.shape
    h = random.randint(int(img.shape[2] / 4), img.shape[2] - int(img.shape[2] / 4))
    w = random.randint(int(img.shape[3] / 4), img.shape[3] - int(img.shape[3] / 4))
    img = torch.nn.functional.interpolate(img, (h, w), mode='bilinear')
    new_img2 = torch.zeros(im_shape).to(img)
    i = random.randint(0, im_shape[2] - h)
    j = random.randint(0, im_shape[2] - w)
    new_img2[:, :, i:i + h, j:j + w] = img

    factor_h = h / (im_shape[-1] - 1)
    factor_w = w / (im_shape[-2] - 1)
    i = i/(im_shape[-1] - 1)
    j = j/(im_shape[-2] - 1)
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes[..., 0] = boxes[..., 0]*factor_w + j
    boxes[..., 1] = boxes[..., 1]*factor_h + i
    boxes[..., 2] = boxes[..., 2]*factor_w + j
    boxes[..., 3] = boxes[..., 3]*factor_h + i
    boxes = box_xyxy_to_cxcywh(boxes)
    img = new_img2
    return img, boxes


def h_flip(x, y):
    y = y.clone()
    x = TF.hflip(x)
    y[..., 0] = 1 - y[..., 0]
    return x, y


def v_flip(x, y):
    y = y.clone()
    x = TF.vflip(x)
    y[..., 1] = 1 - y[..., 1]
    return x, y

def random_translate(image, mask, box_scale, box_shift):
    im_shape = image.shape
    h = random.randint(int(im_shape[1] / 4), im_shape[1] - int(im_shape[1] / 4))
    w = random.randint(int(im_shape[2] / 4), im_shape[2] - int(im_shape[2] / 4))
    img = torch.nn.functional.interpolate(image.unsqueeze(0), (h, w), mode='bilinear')[0]
    m = torch.nn.functional.interpolate(mask.unsqueeze(0).type(torch.float), (h, w), mode='nearest')[0].type(torch.bool)
    new_img2 = torch.zeros(im_shape).to(img)
    new_mask2 = torch.zeros(mask.shape).to(mask)
    i = random.randint(0, im_shape[1] - h)
    j = random.randint(0, im_shape[2] - w)
    new_img2[:, i:i + h, j:j + w] = img
    new_mask2[:, i:i + h, j:j + w] = m
    factor_h = h / (im_shape[1] - 1)
    factor_w = w / (im_shape[2] - 1)
    i = i / (im_shape[1] - 1)
    j = j / (im_shape[2] - 1)
    box_scale[0::2] *= factor_w
    box_scale[1::2] *= factor_h
    box_shift[0] = box_shift[0]*factor_w + j
    box_shift[1] = box_shift[1]*factor_h + i
    return new_img2, new_mask2, box_scale, box_shift

@torch.no_grad()
def get_random_image_and_perm(image, mask):
    box_scale = torch.ones((4))
    box_shift = torch.zeros((4))
    new_img2 = image.clone()
    new_mask2 = mask.clone()
    new_mask2 = new_mask2.unsqueeze(0)
    new_img2, new_mask2, box_scale, box_shift = random_crop_and_resize(new_img2, new_mask2, box_scale, box_shift)

    if random.random() > 0.5:
        new_img2, new_mask2, box_scale, box_shift = random_translate(new_img2, new_mask2, box_scale, box_shift)

    if random.random() > 0.5:
        new_img2 = TF.hflip(new_img2)
        new_mask2 = TF.hflip(new_mask2)
        box_shift[0] = 1 - box_shift[0]
        box_scale[0] *= -1
    if random.random() > 0.5:
        new_img2 = TF.vflip(new_img2)
        new_mask2 = TF.vflip(new_mask2)
        box_shift[1] = 1 - box_shift[1]
        box_scale[1] *= -1

    box_trans = torch.stack([box_scale, box_shift], dim=-1)
    return box_trans, new_img2, new_mask2[0]


def random_crop_and_resize(new_img2, new_mask2, box_scale, box_shift):
    box_scale = box_scale.clone()
    box_shift = box_shift.clone()
    i, j, h, w = RandomResizedCrop.get_params(new_img2, scale=(0.5, 1.), ratio=(1. / 4., 4. / 3.))
    orig_w = new_img2.shape[2]
    orig_h = new_img2.shape[1]
    new_img2 = TF.crop(new_img2, i, j, h, w)
    new_mask2 = TF.crop(new_mask2, i, j, h, w)
    new_img2 = torch.nn.functional.interpolate(new_img2.unsqueeze(0), (orig_h, orig_w), mode='bilinear')[0]
    new_mask2 = torch.nn.functional.interpolate(new_mask2.unsqueeze(0).type(torch.float), (orig_h, orig_w), mode='nearest')[0].type(torch.bool)
    i = i / (orig_h - 1)
    j = j / (orig_w - 1)
    f_h = orig_h / h
    f_w = orig_w / w
    box_shift[1] = (box_shift[1] - i) * f_h
    box_shift[0] = (box_shift[0] - j) * f_w
    box_scale[1] *= box_scale[1] * f_h
    box_scale[0] *= box_scale[0] * f_w
    box_scale[3] *= f_h
    box_scale[2] *= f_w
    return new_img2, new_mask2, box_scale, box_shift
