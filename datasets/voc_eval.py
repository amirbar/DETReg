import os
import shutil
import datetime
import functools
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import torch

from util.misc import all_gather

class VocEvaluator:
    def __init__(self, voc_gt, iou_types, use_07_metric = True, ovthresh = list(range(50, 100, 5))):
        assert tuple(iou_types) == ('bbox', )
        self.use_07_metric = use_07_metric
        self.ovthresh = ovthresh

        self.voc_gt = voc_gt
        self.eps = torch.finfo(torch.float64).eps
        self.num_classes = len(self.voc_gt.CLASS_NAMES)
        self.AP = torch.zeros(self.num_classes, len(ovthresh))
        self.coco_eval = dict(bbox = lambda: None)
        self.coco_eval['bbox'].stats = torch.tensor([])
        self.coco_eval['bbox'].eval = dict()
        
        self.img_ids = []
        self.lines = []
        self.lines_cls = []
    
    def update(self, predictions):
        for img_id, pred in predictions.items():
            pred_boxes, pred_labels, pred_scores = [pred[k].cpu() for k in ['boxes', 'labels', 'scores']]
            image_id = self.voc_gt.convert_image_id(int(img_id), to_string = True)
            self.img_ids.append(img_id)

            for (xmin, ymin, xmax, ymax), cls, score in zip(pred_boxes.tolist(), pred_labels.tolist(), pred_scores.tolist()):
                xmin += 1
                ymin += 1
                self.lines.append(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)
        
    def synchronize_between_processes(self):
        self.img_ids = torch.tensor(self.img_ids, dtype = torch.int64)
        self.lines_cls = torch.tensor(self.lines_cls, dtype = torch.int64)
        self.img_ids, self.lines, self.lines_cls = self.merge(self.img_ids, self.lines, self.lines_cls)

    def merge(self, img_ids, lines, lines_cls):
        flatten = lambda ls: [s for l in ls for s in l]
        
        all_img_ids = torch.cat(all_gather(img_ids))
        all_lines_cls = torch.cat(all_gather(lines_cls))
        all_lines = flatten(all_gather(lines))
        
        # keep only unique (and in sorted order) images
        #merged_img_ids, idx = np.unique(all_img_ids.numpy(), return_index=True); merged_img_ids, idx = torch.as_tensor(merged_img_ids), torch.as_tensor(idx);
        
        #merged_lines_cls = all_lines_cls[idx]
        #merged_lines = [all_lines[i] for i in idx.tolist()]

        return all_img_ids, all_lines, all_lines_cls
        #return merged_img_ids, merged_lines, merged_lines_cls

    def accumulate(self):
        for class_label_ind, class_label in enumerate(self.voc_gt.CLASS_NAMES):
            lines_by_class = [l + '\n' for l, c in zip(self.lines, self.lines_cls.tolist()) if c == class_label_ind]
            for ovthresh_ind, ovthresh in enumerate(self.ovthresh):
                self.AP[class_label_ind, ovthresh_ind] = voc_eval(lines_by_class, self.voc_gt.annotations, self.voc_gt.image_set, class_label, ovthresh = ovthresh / 100.0, use_07_metric = self.use_07_metric)[-1]
                
       
        self.eval = dict(params = dict(usef_07_metric = self.use_07_metric, ovthresh = self.ovthresh), date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), AP = self.AP)

    def summarize(self, fmt = '{:.06f}'):
        o50, o75 = map(self.ovthresh.index, [50, 75])
        mAP = float(self.AP.mean())
        mAP50 = float(self.AP[:, o50].mean())
        mAP75 = float(self.AP[:, o75].mean())
        print('detection mAP50:', fmt.format(mAP50))
        print('detection mAP75:', fmt.format(mAP75))
        print('detection mAP:', fmt.format(mAP))
        print('---AP50---')
        for class_name, ap in zip(self.voc_gt.CLASS_NAMES, self.AP[:, o50].cpu().tolist()):
            print(class_name, fmt.format(ap))
        self.coco_eval['bbox'].stats = torch.cat([self.AP[:, o50].mean(dim = 0, keepdim = True), self.AP[:, o75].mean(dim = 0, keepdim = True), self.AP.flatten().mean(dim = 0, keepdim = True), self.AP.flatten()])
    


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

@functools.lru_cache(maxsize = None)
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects



def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    
    # --------------------------------------------------------
    # Fast/er R-CNN
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Bharath Hariharan
    # --------------------------------------------------------

    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    def iou(BBGT, bb):
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        return ovmax, jmax
    
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # read list of images
    if isinstance(imagesetfile, list):
        lines = imagesetfile
    else:
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    if isinstance(annopath, list):
        for a in annopath:
            imagename = os.path.splitext(os.path.basename(a))[0]
            recs[imagename] = parse_rec(a)
    else:
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    if isinstance(detpath, list):
        lines = detpath
    else:
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
    if BB.size == 0:
        return 0, 0, 0

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ovmax, jmax = iou(BBGT, bb)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def bbox_nms(boxes, scores, overlap_threshold = 0.4, score_threshold = 0.0, mask = False):
    
    def overlap(box1, box2 = None, rectint = False, eps = 1e-6):
        area = lambda boxes = None, x1 = None, y1 = None, x2 = None, y2 = None: (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]) if boxes is not None else (x2 - x1).clamp(min = 0) * (y2 - y1).clamp(min = 0)

        if box2 is None and not isinstance(box1, list) and box1.dim() == 3:
            return torch.stack(list(map(overlap, box1)))
        b1, b2 = [(b if b.dim() == 2 else b.unsqueeze(0)).t().contiguous() for b in [box1, (box2 if box2 is not None else box1)]]

        xx1 = torch.max(b1[0].unsqueeze(1), b2[0].unsqueeze(0))
        yy1 = torch.max(b1[1].unsqueeze(1), b2[1].unsqueeze(0))
        xx2 = torch.min(b1[2].unsqueeze(1), b2[2].unsqueeze(0))
        yy2 = torch.min(b1[3].unsqueeze(1), b2[3].unsqueeze(0))
        
        inter = area(x1 = xx1, y1 = yy1, x2 = xx2, y2 = yy2)
        return inter / (area(b1.t()).unsqueeze(1) + area(b2.t()).unsqueeze(0) - inter + eps) if not rectint else inter

    O = overlap(boxes)
    I = scores.sort(0)[1]
    M = scores.gather(0, I).ge(score_threshold)
    M = M if M.any() else M.fill_(1)
    pick = []
    
    for i, m in zip(I.t(), M.t()):
        p = []
        i = i[m]
        while len(i) > 1:
            p.append(i[-1])
            m = O[:, i[-1]][i].lt(overlap_threshold)
            m[-1] = 0
            i = i[m]
        pick.append(torch.tensor(p + i.tolist(), dtype = torch.int64))

    return pick if not mask else torch.stack([torch.zeros(len(scores), dtype = torch.bool).scatter_(0, p, 1) for p in pick])

def package_submission(out_dir, image_file_name, class_labels, VOCYEAR, SUBSET, TASK, tar = True, **kwargs):
    def cls(file_path, class_label_ind, scores):
        with open(file_path, 'w') as f:
            f.writelines(map('{} {}\n'.format, image_file_name, scores[:, class_label_ind].tolist()))

    def det(file_path, class_label_ind, scores, proposals, keep):
        zipped = []
        for example_idx, basename in enumerate(image_file_name):
            I = keep[example_idx][class_label_ind]
            zipped.extend((basename, s) + tuple(p) for s, p in zip(scores[example_idx][I, class_label_ind].tolist(), proposals[example_idx][I, :4].add(1).tolist()))
        with open(file_path, 'w') as f:
            f.writelines(map('{} {} {:.0f} {:.0f} {:.0f} {:.0f} \n'.format, *zip(*zipped)))

    task_a, task_b = TASK.split('_')
    resdir = os.path.join(out_dir, 'results')
    respath = os.path.join(resdir, VOCYEAR, 'Main', '%s_{}_{}_%s.txt'.format(task_b, SUBSET))
    
    if os.path.exists(resdir):
        shutil.rmtree(resdir)
    os.makedirs(os.path.join(resdir, VOCYEAR, 'Main'))
    
    for class_label_ind, class_label in enumerate(class_labels):
        dict(det = det, cls = cls)[task_b](respath.replace('%s', '{}').format(task_a, class_label), class_label_ind, **kwargs)
    
    if tar:
        subprocess.check_call(['tar', '-czf', 'results-{}-{}-{}.tar.gz'.format(VOCYEAR, TASK, SUBSET), 'results'], cwd = out_dir)
    
    return respath

def detection_mean_ap(out_dir, image_file_name, class_labels, VOCYEAR, SUBSET, VOC_DEVKIT_VOCYEAR, scores = None, boxes = None, nms_score_threshold = 1e-4, nms_overlap_threshold = 0.4, tar = False, octave = False, cmd = 'octave --eval', env = None, stdout_stderr = open(os.devnull, 'wb'), do_nms = True):
    
    if scores is not None:
        nms = list(map(lambda s, p: bbox_nms(p, s, overlap_threshold = nms_overlap_threshold, score_threshold = nms_score_threshold), scores,  boxes )) if do_nms else [torch.arange(len(p)) for p in boxes]

    else:
        nms =  torch.arange(len(class_labels)).unsqueeze(0).unsqueeze(-1).expand(len(image_file_name), len(class_labels), 1)
        scores = torch.zeros(len(image_file_name), len(class_labels), len(class_labels))

    imgsetpath = os.path.join(VOC_DEVKIT_VOCYEAR, 'ImageSets', 'Main', SUBSET + '.txt')
    detrespath = package_submission(out_dir, image_file_name, class_labels, VOCYEAR, SUBSET, 'comp4_det', tar = tar, scores = scores, proposals = boxes, nms = nms)

    if octave:
        imgsetpath_fix = os.path.join(out_dir, detection_mean_ap.__name__ + '.txt')
        with open(imgsetpath_fix, 'w') as f:
            f.writelines([line[:-1] + ' -1\n' for line in open(imgsetpath)])
        procs = [subprocess.Popen(cmd.split() + ["oldpwd = pwd; cd('{}/..'); addpath(fullfile(pwd, 'VOCcode')); VOCinit; cd(oldpwd); VOCopts.testset = '{}'; VOCopts.detrespath = '{}'; VOCopts.imgsetpath = '{}'; classlabel = '{}'; warning('off', 'Octave:possible-matlab-short-circuit-operator'); warning('off', 'Octave:num-to-str'); [rec, prec, ap] = VOCevaldet(VOCopts, 'comp4', classlabel, false); dlmwrite(sprintf(VOCopts.detrespath, 'resu4', classlabel), ap); quit;".format(VOC_DEVKIT_VOCYEAR, SUBSET, detrespath, imgsetpath_fix, class_label)], stdout = stdout_stderr, stderr = stdout_stderr, env = env) for class_label in class_labels]
        res = list(map(lambda class_label, proc: proc.wait() or float(open(detrespath % ('resu4', class_label)).read()), class_labels, procs))
    
    else:
        res = [voc_eval(detrespath.replace('%s', '{}').format('comp4', '{}'), os.path.join(VOC_DEVKIT_VOCYEAR, 'Annotations', '{}.xml'), imgsetpath, class_label, cachedir = os.path.join(out_dir, 'cache_detection_mean_ap_' + SUBSET), use_07_metric = True)[-1] for class_label in class_labels]

    return torch.tensor(res).mean(), res
