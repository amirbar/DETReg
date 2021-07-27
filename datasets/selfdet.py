# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
pre-training dataset which implements random query patch detection.
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
import datasets.transforms as T
from torchvision.transforms import transforms
from PIL import ImageFilter
import random
import cv2
from util.box_ops import crop_bbox


def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh


class SelfDet(Dataset):
    """
    SelfDet is a dataset class which implements random query patch detection.
    It randomly crops patches as queries from the given image with the corresponding bounding box.
    The format of the bounding box is same to COCO.
    """

    def __init__(self, root, detection_transform, query_transform, cache_dir=None, max_prop=30, strategy='topk'):
        super(SelfDet, self).__init__()
        self.strategy = strategy
        self.cache_dir = cache_dir
        self.query_transform = query_transform
        self.root = root
        self.max_prop = max_prop
        self.detection_transform = detection_transform
        self.files = []
        self.dist2 = -np.log(np.arange(1, 301) / 301) / 10
        max_prob = (-np.log(1 / 1001)) ** 4

        for (troot, _, files) in os.walk(root, followlinks=True):
            for f in files:
                if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                    path = os.path.join(troot, f)
                    self.files.append(path)
                else:
                    continue
        print(f'num of files:{len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_path = self.files[item]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if self.strategy == 'topk':
            boxes = selective_search(img, h, w, res_size=128)
            boxes = boxes[:self.max_prop]
        elif self.strategy == 'mc':
            boxes = self.get_boxes_from_ss_or_cache(self.dist2, h, img, item, w)
        elif self.strategy == "random":
            fn = random.choice(self.files).split('/')[-1].split('.')[0] + '.npy'
            fp = os.path.join(self.cache_dir, fn)
            try:
                with open(fp, 'rb') as f:
                    boxes = np.load(f)
            except FileNotFoundError:
                boxes = selective_search(img, h, w, res_size=None)
                with open(fp, 'wb') as f:
                    np.save(f, boxes)
            boxes = boxes[:self.max_prop]
        else:
            raise ValueError("No such strategy")

        # from util.plot_utils import plot_opencv, plot_results
        # from matplotlib import pyplot as plt
        # output = np.array(img)
        # plot_opencv(boxes[:1], output)
        # plt.figure()
        # plot_results(np.array(img), np.zeros(1), boxes[:1], plt.gca(), norm=False)
        # plt.show()
        # plt.figure()
        # plt.imshow(patches[0])

        if len(boxes) < 2:
            return self.__getitem__(random.randint(0, len(self.files) - 1))

        patches = [img.crop([b[0], b[1], b[2], b[3]]) for b in boxes]
        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        target['patches'] = torch.stack([self.query_transform(p) for p in patches], dim=0)
        target['boxes'] = torch.tensor(boxes)
        target['iscrowd'] = torch.zeros(len(target['boxes']))
        target['area'] = target['boxes'][..., 2] * target['boxes'][..., 3]
        target['labels'] = torch.ones(len(target['boxes'])).long()
        img, target = self.detection_transform(img, target)
        if len(target['boxes']) < 2:
            return self.__getitem__(random.randint(0, len(self.files) - 1))
        # crop_size = 96  # TODO: add as hyperparam
        # target['patches'] = crop_bbox(img.unsqueeze(0).repeat_interleave(len(target['boxes']), 0), target['boxes'],
        #                               crop_size, crop_size)

        # from matplotlib import pyplot as plt
        # plot_prediction(img.unsqueeze(0), target['boxes'][:1].unsqueeze(1), torch.zeros(1, 1, 2), plot_prob=False)
        # plt.show()
        # plot_prediction(target['patches'][0].unsqueeze(0), torch.zeros_like(target['boxes'][:1].unsqueeze(1)), torch.zeros(1, 1, 2), plot_prob=False)
        # plt.show()
        return img, target

    def get_boxes_from_ss_or_cache(self, func, h, img, item, w):
        fn = self.files[item].split('/')[-1].split('.')[0] + '.npy'
        fp = os.path.join(self.cache_dir, fn)
        try:
            with open(fp, 'rb') as f:
                boxes = np.load(f)
        except FileNotFoundError:
            boxes = selective_search(img, h, w, res_size=None)
            with open(fp, 'wb') as f:
                np.save(f, boxes)
        boxes_indicators = np.where(np.random.binomial(1, p=func[:len(boxes)]))[0]
        boxes = boxes[boxes_indicators]
        return boxes


def selective_search(img, h, w, res_size=128):
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('float32')

    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


def make_self_det_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # The image of ImageNet is relatively small.
    scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=600),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=600),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=600),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_query_transforms(image_set):
    if image_set == 'train':
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if image_set == 'val':
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    raise ValueError(f'unknown {image_set}')


def build_selfdet(image_set, args, p):
    return SelfDet(p, detection_transform=make_self_det_transforms(image_set), query_transform=get_query_transforms(image_set), cache_dir=args.cache_path,
                   max_prop=args.max_prop, strategy=args.strategy)
