# coding: utf-8

from __future__ import division, print_function

import numpy as np
import torch
from utils.GIOU import iou_calc3
import einops

def torch_nms_sampling(cfg, boxes, variance=None):
    def nms_class(clsboxes, clsscores):
        keep = []
        keepscores = []
        while clsboxes.shape[0] > 0:
            maxidx = torch.argmax(clsscores)
            maxbox = clsboxes[maxidx].unsqueeze(0)
            clsboxes = torch.cat((clsboxes[:maxidx], clsboxes[maxidx + 1:]), 0)
            keepscores.append(clsscores[maxidx])
            clsscores = torch.cat((clsscores[:maxidx], clsscores[maxidx + 1:]), 0)
            iou = iou_calc3(maxbox[:, :4, 0], clsboxes[:, :4, 0])
            keep.append(maxbox)

            weight = torch.ones_like(iou)
            if not cfg.soft:
                weight[iou > cfg.nms_iou] = 0
            else:
                weight = torch.exp(-1.0 * (iou ** 2 / cfg.softsigma))
            clsscores = clsscores * weight
            filter_idx = torch.nonzero(clsscores >= cfg.score_thres, as_tuple = False).squeeze(-1)
            clsboxes = clsboxes[filter_idx]
            clsscores = clsscores[filter_idx]
        return torch.cat(keep, 0).to(clsboxes.device), torch.stack(keepscores, 0).to(clsscores.device)

    bbox = boxes[:, :4, :].view(-1, 4, boxes.shape[2])
    numcls = boxes.shape[1] - 4
    scores = boxes[:, 4:, 0].view(-1, numcls)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(numcls):
        filter_idx = torch.nonzero(scores[:, i] >= cfg.score_thres, as_tuple = False).squeeze(-1)
        if len(filter_idx) == 0:
            continue
        filter_boxes = bbox[filter_idx, :, :]
        filter_scores = scores[:, i][filter_idx]
        clsbox, clsscores = nms_class(filter_boxes, filter_scores)
        if clsbox.shape[0] > 0:
            picked_boxes.append(clsbox)
            picked_score.append(clsscores)
            picked_label.extend([torch.ByteTensor([i]) for _ in range(len(clsbox))])
    if len(picked_boxes) == 0:
        return None, None, None
    else:
        return torch.cat(picked_boxes), torch.cat(picked_score), torch.cat(picked_label)
