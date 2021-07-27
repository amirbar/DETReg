# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy
import numpy as np
from torchvision.ops.boxes import box_area

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # [N,M,2]
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [N,M]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all()
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[..., :2], boxes2[...,:2])
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[..., 0] * wh[..., 1]

    return iou - (area - union) / area

def return_matching(cost):
    indices = np.array(list(map(linear_sum_assignment, cost.detach().cpu().numpy())))
    indices = np.transpose(indices, axes=(0, 2, 1))[..., 1:]
    indices = torch.LongTensor(indices).to(cost).type(torch.int64)
    return indices


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, out_bbox, tgt_bbox):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries1 = out_bbox.shape[:2]
            _, num_queries2 = tgt_bbox.shape[:2]
            # We flatten to compute the cost matrices in a batch
            # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            # out_bbox = out_bbox.flatten(0, 1)
            # Also concat the target labels and boxes
            # tgt_ids = torch.cat([v["labels"] for v in targets])
            # tgt_bbox = tgt_bbox.flatten(0, 1)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            out_bbox1 = out_bbox.unsqueeze(2).repeat_interleave(num_queries2, 2)
            tgt_bbox2 = tgt_bbox.unsqueeze(1).repeat_interleave(num_queries1, 1)
            cost_bbox = torch.nn.functional.l1_loss(out_bbox1, tgt_bbox2, reduction='none').mean(dim=-1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox1.flatten(0, 2)), box_cxcywh_to_xyxy(tgt_bbox2.flatten(0, 2)))
            cost_giou = cost_giou.view(bs, num_queries1, num_queries2)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            indices = return_matching(C)
            # identity matching
            indices = torch.arange(0, C.shape[1]).view(1, C.shape[1], 1).expand_as(indices).to(indices)
        return indices


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
