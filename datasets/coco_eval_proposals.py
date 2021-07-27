# pip install pycocotools opencv-python opencv-contrib-python
# wget https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz


import os
import copy
import time
import argparse
import contextlib
import multiprocessing

import numpy as np
import cv2
import cv2.ximgproc

import matplotlib.patches
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def imshow_with_boxes(img, boxes_xywh, savefig):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    for x, y, w, h in boxes_xywh.tolist():
        plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.savefig(savefig)
    plt.close()
    return savefig

def selective_search(img, fast, topk):
    algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    algo.setBaseImage(img)
    if fast:
        algo.switchToSelectiveSearchFast()
    else:
        algo.switchToSelectiveSearchQuality()

    boxes_xywh = algo.process().astype(np.float32)
    scores = np.ones( (len(boxes_xywh), ) )
    return boxes_xywh[:topk], scores[:topk]

def edge_boxes(img, fast, topk, bgr2rgb = (2, 1, 0), algo_edgedet = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz') if os.path.exists('model.yml.gz') else None):
    edges = algo_edgedet.detectEdges(img[..., bgr2rgb].astype(np.float32) / 255.0)
    orimap = algo_edgedet.computeOrientation(edges)
    edges = algo_edgedet.edgesNms(edges, orimap)
    algo_edgeboxes = cv2.ximgproc.createEdgeBoxes()
    algo_edgeboxes.setMaxBoxes(topk)
    boxes_xywh, scores = algo_edgeboxes.getBoundingBoxes(edges, orimap)
    
    if scores is None:
        boxes_xywh, scores = np.array([[0, 0.0, img.shape[1], img.shape[0]]]), np.ones((1, ))
    
    return boxes_xywh, scores.squeeze()

def process_image(image_id, img_extra, fast, resize, algo, rgb2bgr = (2, 1, 0), category_other = -1, topk = 1000):
    img = np.asarray(img_extra[0])[..., rgb2bgr]
    h, w = img.shape[:2]
    
    img_det = img if resize == 1 else cv2.resize(img, (resize, resize))
   
    boxes_xywh, scores = algo(img_det, fast, topk)

    boxes_xywh = boxes_xywh.astype(np.float32) * (1 if resize == 1 else np.array([w, h, w, h]) / resize)

    labels = np.full((len(boxes_xywh), ), category_other, dtype = int)
    
    return image_id, dict(boxes = boxes_xywh, scores = scores, labels = labels)

def process_loaded(image_id, loaded, category_other = -1):
    boxes_xyxy = loaded['pred_boxes_'].clamp(min = 0)
   
    boxes_xywh = torch.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]], dim = -1)

    labels = np.full((len(boxes_xywh), ), category_other, dtype = int)
   
    num_classes = loaded['pred_logits'].shape[-1]
    scores = loaded['pred_logits'][:, 1:: num_classes - 2][:, 0]
    
    I = scores.argsort(descending = True)

    scores = scores[I]
    boxes_xywh = boxes_xywh[I]
    labels = labels[I]
    
    return image_id, dict(boxes = boxes_xywh, scores = scores, labels = labels)
    

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_type = 'bbox', useCats = 0, maxDets = 100):
        self.coco_gt = copy.deepcopy(coco_gt)
        self.coco_eval = COCOeval(coco_gt, iouType = iou_type)
        if maxDets != [100]:
            self.coco_eval.params.maxDets = maxDets
        if not useCats:
            self.coco_eval.params.useCats = useCats
            self.coco_eval.params.catIds = [-1]
            coco_gt.loadAnns = lambda imgIds, loadAnns = coco_gt.loadAnns: [gt.update(dict(category_id = -1)) or gt for gt in loadAnns(imgIds)] 
        self.accumulate, self.summarize = self.coco_eval.accumulate, self.coco_eval.summarize

    @staticmethod
    def call_without_stdout(func, *args):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*args)

    def update(self, predictions):
        tolist = lambda a: [a.tolist()] if a.ndim == 0 else a.tolist()

        detection_results = [dict(image_id = image_id, bbox = bbox, score = score, category_id = category_id) for image_id, pred in predictions.items() if pred for bbox, score, category_id in zip(pred['boxes'].tolist(), tolist(pred['scores']), pred['labels'].tolist())]
        self.coco_eval.cocoDt = self.call_without_stdout(COCO.loadRes, self.coco_gt, detection_results) if detection_results else COCO()
        self.coco_eval.params.imgIds = list(predictions)
        self.call_without_stdout(self.coco_eval.evaluate)


def main(args):
    coco_mode = 'instances'
    PATHS = dict(
        train = (os.path.join(args.dataset_root, f'train{args.dataset_year}'), os.path.join(args.dataset_root, 'annotations', f'{coco_mode}_train{args.dataset_year}.json')),
        val = (os.path.join(args.dataset_root, f'val{args.dataset_year}'), os.path.join(args.dataset_root, 'annotations', f'{coco_mode}_val{args.dataset_year}.json')),
    )
    dataset = CocoDetection(*PATHS[args.dataset_split]) 
    coco_evaluator = CocoEvaluator(dataset.coco, maxDets = args.max_dets)
    
    tic = time.time()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok = True)

    if args.algo != 'process_loaded':
        preds = dict(multiprocessing.Pool(processes = args.num_workers).starmap(process_image, zip(dataset.ids, dataset, [args.fast] * len(dataset), [args.resize] * len(dataset), [globals()[args.algo]] * len(dataset))))

    else:
        preds = []
        for i, t in enumerate(zip(dataset.ids, dataset, [args.fast] * len(dataset),  [args.resize] * len(dataset), [globals()[args.algo]] * len(dataset))):
            loaded = torch.load(os.path.join(args.input_dir, str(t[0]) + '.pt'), map_location = 'cpu')
            preds.append(process_loaded(t[0], loaded))
            if args.output_dir:
                imshow_with_boxes(t[1][0], preds[-1][1]['boxes'][:5], os.path.join(args.output_dir, str(t[0]) + '.jpg'))
            print(i) if i % 50 == 0 else None

    preds = dict(preds)
    
    print('proposals', time.time() - tic); tic = time.time()
    coco_evaluator.update(preds)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print('evaluator', time.time() - tic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i')
    parser.add_argument('--output-dir', '-o')
    parser.add_argument('--dataset-root')
    parser.add_argument('--dataset-split', default = 'val', choices = ['train', 'val'])
    parser.add_argument('--dataset-year', type = int, default = 2017)
    parser.add_argument('--num-workers', type = int, default = 16)
    parser.add_argument('--algo', default = 'selective_search', choices = ['selective_search', 'edge_boxes', 'process_loaded'])
    parser.add_argument('--fast', action = 'store_true')
    parser.add_argument('--resize', type = int, default = 128)
    parser.add_argument('--max-dets', type = int, nargs = '*', default = [100])
    
    args = parser.parse_args()
    print(args)
    main(args)
