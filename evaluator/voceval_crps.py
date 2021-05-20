from utils.visualize import visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.dataset_util import PascalVocXmlParser
from collections import defaultdict
import os
from .Evaluator import Evaluator
import csv
import math
import pickle
class EvaluatorVOC_CRPS(Evaluator):
    def __init__(self, anchors, cateNames, rootpath, score_thres, iou_thres, use_07_metric=False):
        self.rec_pred = defaultdict(list)
        self.rec_gt = defaultdict(list)
        self.use_07_metric = use_07_metric
        self._annopath = os.path.join(rootpath, 'VOC2007', 'Annotations', '{}.xml')
        self._imgpath = os.path.join(rootpath, 'VOC2007', 'JPEGImages', '{}.jpg')
        self.reset()
        super().__init__(anchors, cateNames, rootpath, score_thres, iou_thres)

    def reset(self):
        self.coco_imgIds = set([])
        self.visual_imgs = []
        self.rec_pred = defaultdict(list)

    def append(self, imgpath, nms_boxes, nms_scores, nms_labels, variance, visualize=True):
        if nms_boxes is not None:  # do have bboxes
            for i in range(nms_boxes.shape[0]):
                rec = {
                    "img_idx": imgpath.split('/')[-1].split('.')[0],
                    "bbox": nms_boxes[i],
                    "score": float(nms_scores[i]),
                    "vari": variance[i]
                }
                self.rec_pred[nms_labels[i]].append(rec)
            if visualize and len(self.visual_imgs) < self.num_visual:
                # _, boxGT, labelGT, _ = PascalVocXmlParser(str(annpath), self.cateNames).parse()
                # boxGT=np.array(boxGT)
                # labelGT=np.array(labelGT)
                # self.append_visulize(imgpath, nms_boxes, nms_labels, nms_scores, boxGT, labelGT)
                self.append_visulize(imgpath, nms_boxes, nms_labels, nms_scores, None, None)

    def save_dict(self):
        pickle.dump(self.rec_pred, open( "rec_pred.p", "wb" ) )
    def load_dict(self):
        self.rec_pred = pickle.load(open( "rec_pred.p", "rb" ))
    def evaluate(self):
        crps_per_class = []
        for idx, cls in enumerate(self.cateNames):
            crps_per_square = []
            box_samples = {}
            if len(self.rec_pred[idx]) > 0:
                _recs_pre = self.rec_pred[idx]
                num_recs_pre = len(_recs_pre)
                scores = np.array([rec['score'] for rec in _recs_pre])
                sorted_ind = np.argsort(-scores)
                scores = scores[sorted_ind]
                bboxs = np.array([rec['bbox'] for rec in _recs_pre])[sorted_ind]
                img_idxs = [rec['img_idx'] for rec in _recs_pre]
                img_idxs = [img_idxs[idx] for idx in sorted_ind]

                # build recgt according to appeard imgs
                _recs_gt = defaultdict(dict)
                for imgidx in set(img_idxs):
                    _rec = [rec for rec in self.rec_gt[imgidx[19:]] if rec['label'] == self.cateNames.index(cls)]
                    _box = np.array([rec['bbox'] for rec in _rec])
                    _dif = np.array([rec['difficult'] for rec in _rec]).astype(np.bool)
                    _recs_gt[imgidx]['bbox'] = _box
                    _recs_gt[imgidx]['difficult'] = _dif

                # computer iou for each pred record
                for idx in range(len(img_idxs)):
                    _rec_gt = _recs_gt[img_idxs[idx]]
                    _bbGT = _rec_gt['bbox']
                    _bbPre = bboxs[idx, :, 0]
                    ovmax = -np.inf

                    if _bbGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(_bbGT[:, 0], _bbPre[0])
                        iymin = np.maximum(_bbGT[:, 1], _bbPre[1])
                        ixmax = np.minimum(_bbGT[:, 2], _bbPre[2])
                        iymax = np.minimum(_bbGT[:, 3], _bbPre[3])

                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((_bbPre[2] - _bbPre[0]) * (_bbPre[3] - _bbPre[1]) +
                               (_bbGT[:, 2] - _bbGT[:, 0]) *
                               (_bbGT[:, 3] - _bbGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    # TODO add flexible threshold
                    if ovmax > self.iou_thres:
                        if not _rec_gt['difficult'][jmax]:
                            if img_idxs[idx] in box_samples:
                                if str(jmax) in box_samples[img_idxs[idx]]:
                                    box_samples[img_idxs[idx]][str(jmax)].append(bboxs[idx, :, 1:])
                                else:
                                    box_samples[img_idxs[idx]][str(jmax)] = [bboxs[idx, :, 1:]]
                            else:
                                box_samples[img_idxs[idx]] = {}
                                box_samples[img_idxs[idx]][str(jmax)] = [bboxs[idx, :, 1:]]
                #compute CRPS
                for img_idx in box_samples:
                    for img_num in box_samples[img_idx]:
                        samples = np.concatenate(box_samples[img_idx][img_num], axis = 0).transpose().reshape(4, -1)
                        num_samples = samples.shape[1]
                        crps_per_square.append(self.crps_sampling(samples, _recs_gt[img_idx]["bbox"][int(img_num), :], num_samples))

            crps_per_class.append(np.mean(np.array(crps_per_square)))
        return crps_per_class
    def get_distribution_samples(self):
        dist = []
        for idx, cls in enumerate(self.cateNames):
            crps_per_square = []
            box_samples = {}
            if len(self.rec_pred[idx]) > 0:
                _recs_pre = self.rec_pred[idx]
                num_recs_pre = len(_recs_pre)
                scores = np.array([rec['score'] for rec in _recs_pre])
                sorted_ind = np.argsort(-scores)
                scores = scores[sorted_ind]
                bboxs = np.array([rec['bbox'] for rec in _recs_pre])[sorted_ind]
                variance = np.array([rec['vari'] for rec in _recs_pre])[sorted_ind]
                img_idxs = [rec['img_idx'] for rec in _recs_pre]
                img_idxs = [img_idxs[idx] for idx in sorted_ind]

                # build recgt according to appeard imgs
                _recs_gt = defaultdict(dict)
                for imgidx in set(img_idxs):
                    _rec = [rec for rec in self.rec_gt[imgidx[19:]] if rec['label'] == self.cateNames.index(cls)]
                    _box = np.array([rec['bbox'] for rec in _rec])
                    _dif = np.array([rec['difficult'] for rec in _rec]).astype(np.bool)
                    _recs_gt[imgidx]['bbox'] = _box
                    _recs_gt[imgidx]['difficult'] = _dif

                # computer iou for each pred record
                for idx in range(len(img_idxs)):
                    _rec_gt = _recs_gt[img_idxs[idx]]
                    _bbGT = _rec_gt['bbox']
                    _bbPre = bboxs[idx, :, 0]
                    _bbVar = variance[idx, :]
                    ovmax = -np.inf

                    if _bbGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(_bbGT[:, 0], _bbPre[0])
                        iymin = np.maximum(_bbGT[:, 1], _bbPre[1])
                        ixmax = np.minimum(_bbGT[:, 2], _bbPre[2])
                        iymax = np.minimum(_bbGT[:, 3], _bbPre[3])

                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((_bbPre[2] - _bbPre[0]) * (_bbPre[3] - _bbPre[1]) +
                               (_bbGT[:, 2] - _bbGT[:, 0]) *
                               (_bbGT[:, 3] - _bbGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    # TODO add flexible threshold
                    if ovmax > self.iou_thres:
                        if not _rec_gt['difficult'][jmax]:
                            for i in range(len(bboxs[idx, 1, 1:])):
                                dist.append((bboxs[idx, 0, i+1]-_bbPre[0])/math.sqrt(_bbVar[0]))
                                dist.append((bboxs[idx, 1, i+1]-_bbPre[1])/math.sqrt(_bbVar[1]))
                                dist.append((bboxs[idx, 2, i+1]-_bbPre[2])/math.sqrt(_bbVar[2]))
                                dist.append((bboxs[idx, 3, i+1]-_bbPre[3])/math.sqrt(_bbVar[3]))

                #compute CRPS
        dist = np.array(dist)
        np.savetxt("dist_sampling.csv", dist, delimiter=",")
        return None
    def get_distribution(self):
        dist = []
        for idx, cls in enumerate(self.cateNames):
            crps_per_square = []
            box_samples = {}
            if len(self.rec_pred[idx]) > 0:
                _recs_pre = self.rec_pred[idx]
                num_recs_pre = len(_recs_pre)
                scores = np.array([rec['score'] for rec in _recs_pre])
                sorted_ind = np.argsort(-scores)
                scores = scores[sorted_ind]
                bboxs = np.array([rec['bbox'] for rec in _recs_pre])[sorted_ind]
                variance = np.array([rec['vari'] for rec in _recs_pre])[sorted_ind]
                img_idxs = [rec['img_idx'] for rec in _recs_pre]
                img_idxs = [img_idxs[idx] for idx in sorted_ind]

                # build recgt according to appeard imgs
                _recs_gt = defaultdict(dict)
                for imgidx in set(img_idxs):
                    _rec = [rec for rec in self.rec_gt[imgidx[19:]] if rec['label'] == self.cateNames.index(cls)]
                    _box = np.array([rec['bbox'] for rec in _rec])
                    _dif = np.array([rec['difficult'] for rec in _rec]).astype(np.bool)
                    _recs_gt[imgidx]['bbox'] = _box
                    _recs_gt[imgidx]['difficult'] = _dif

                # computer iou for each pred record
                for idx in range(len(img_idxs)):
                    _rec_gt = _recs_gt[img_idxs[idx]]
                    _bbGT = _rec_gt['bbox']
                    _bbPre = bboxs[idx, :, 0]
                    _bbVar = variance[idx, :]
                    ovmax = -np.inf

                    if _bbGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(_bbGT[:, 0], _bbPre[0])
                        iymin = np.maximum(_bbGT[:, 1], _bbPre[1])
                        ixmax = np.minimum(_bbGT[:, 2], _bbPre[2])
                        iymax = np.minimum(_bbGT[:, 3], _bbPre[3])

                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((_bbPre[2] - _bbPre[0]) * (_bbPre[3] - _bbPre[1]) +
                               (_bbGT[:, 2] - _bbGT[:, 0]) *
                               (_bbGT[:, 3] - _bbGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    # TODO add flexible threshold
                    if ovmax > self.iou_thres:
                        if not _rec_gt['difficult'][jmax]:
                            dist.append((_bbGT[jmax, 0]-_bbPre[0])/math.sqrt(_bbVar[0]))
                            dist.append((_bbGT[jmax, 1]-_bbPre[1])/math.sqrt(_bbVar[1]))
                            dist.append((_bbGT[jmax, 2]-_bbPre[2])/math.sqrt(_bbVar[2]))
                            dist.append((_bbGT[jmax, 3]-_bbPre[3])/math.sqrt(_bbVar[3]))

                #compute CRPS
        dist = np.array(dist)
        np.savetxt("dist.csv", dist, delimiter=",")
        return None
    def crps_sampling(self, samples, y_test, num_samples):
        crps_first = (np.mean(np.abs(samples - np.repeat(y_test, num_samples).reshape(-1, num_samples))))
        crps_second = 0
        for i in range(num_samples):
            crps_second = crps_second+ np.mean(np.abs(samples - np.roll(samples, i)))/num_samples
        return crps_first - crps_second/2
    def get_calibration_samples(self):
        calibration_values = []
        for idx, cls in enumerate(self.cateNames):
            box_samples = {}
            if len(self.rec_pred[idx]) > 0:
                num_matched = 0
                _recs_pre = self.rec_pred[idx]
                num_recs_pre = len(_recs_pre)
                scores = np.array([rec['score'] for rec in _recs_pre])
                sorted_ind = np.argsort(-scores)
                scores = scores[sorted_ind]
                bboxs = np.array([rec['bbox'] for rec in _recs_pre])[sorted_ind]
                #print(bboxs.shape)
                calibration_array = np.zeros(len(bboxs[idx, 1, 1:]))
                variance = np.array([rec['vari'] for rec in _recs_pre])[sorted_ind]
                img_idxs = [rec['img_idx'] for rec in _recs_pre]
                img_idxs = [img_idxs[idx] for idx in sorted_ind]

                # build recgt according to appeard imgs
                _recs_gt = defaultdict(dict)
                for imgidx in set(img_idxs):
                    _rec = [rec for rec in self.rec_gt[imgidx[19:]] if rec['label'] == self.cateNames.index(cls)]
                    _box = np.array([rec['bbox'] for rec in _rec])
                    _dif = np.array([rec['difficult'] for rec in _rec]).astype(np.bool)
                    _recs_gt[imgidx]['bbox'] = _box
                    _recs_gt[imgidx]['difficult'] = _dif

                # computer iou for each pred record
                for idx in range(len(img_idxs)):
                    _rec_gt = _recs_gt[img_idxs[idx]]
                    _bbGT = _rec_gt['bbox']
                    _bbPre = bboxs[idx, :, 0]
                    _bbVar = variance[idx, :]
                    ovmax = -np.inf

                    if _bbGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(_bbGT[:, 0], _bbPre[0])
                        iymin = np.maximum(_bbGT[:, 1], _bbPre[1])
                        ixmax = np.minimum(_bbGT[:, 2], _bbPre[2])
                        iymax = np.minimum(_bbGT[:, 3], _bbPre[3])

                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((_bbPre[2] - _bbPre[0]) * (_bbPre[3] - _bbPre[1]) +
                               (_bbGT[:, 2] - _bbGT[:, 0]) *
                               (_bbGT[:, 3] - _bbGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    # TODO add flexible threshold
                    if ovmax > self.iou_thres:
                        if not _rec_gt['difficult'][jmax]:
                            num_matched+=1
                            for i in range(4):
                                #print((_bbGT[jmax, i]<bboxs[idx, i, 1:]))
                                #print(bboxs[idx, i, 1:])
                                calibration_array = calibration_array+(_bbGT[jmax, i]<bboxs[idx, i, 1:])
            #print(calibration_array/num_matched)
            calibration_values.append(calibration_array/(num_matched*4))
        calibration = np.array(calibration_values)
        print(calibration)
        np.savetxt("calibration_metrics.csv", calibration, delimiter=",")
        return None
    def build_GT(self):
        filepath = os.path.join(self.dataset_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
        with open(filepath, 'r') as f:
            filelist = f.readlines()

        filelist = [file.strip() for file in filelist]
        for file in filelist:
            _, boxGT, labelGT, difficult = PascalVocXmlParser(self._annopath.format(file), self.cateNames).parse(
                filterdiff=False)
            for box, label, difficult in zip(boxGT, labelGT, difficult):
                self.rec_gt[file].append({
                    'label': label,
                    'bbox': box,
                    'difficult': difficult
                })

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
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


if __name__ == '__main__':
    dataset_root = '/disk3/datasets/voc'
    _annopath = os.path.join(dataset_root, 'VOC2007', 'Annotations', '{}.xml')
    _imgpath = os.path.join(dataset_root, 'VOC2007', 'JPEGImages', '{}.jpg')
    filepath = os.path.join(dataset_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
    with open(filepath, 'r') as f:
        filelist = f.readlines()
    cateNames = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]
    filelist = [file.strip() for file in filelist]
    rec_gt = defaultdict(list)
    for file in filelist:
        _, boxGT, labelGT, difficult = PascalVocXmlParser(_annopath.format(file), cateNames).parse()
        for box, label, difficult in zip(boxGT, labelGT, difficult):
            rec_gt[file].append({
                'label': label,
                'bbox': box,
                'difficult': difficult
            })
    cls = "person"
    img_idxs = ['000001']
    for imgidx in set(img_idxs):
        print(rec_gt[imgidx])
        _rec = [rec for rec in rec_gt[imgidx] if rec['label'] == cls]
        print(_rec)
