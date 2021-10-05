from utils.util import ensure_dir
from dataset import get_COCO, get_VOC
import os
import time
from dataset import makeImgPyramids
from models.yololoss import yololoss, yololoss_cdf, yololoss_quantile_adjusted, yololoss_cdf_gaussian
from utils.nms_sampling import torch_nms_sampling
from utils.nms_utils import torch_nms
from tensorboardX import SummaryWriter
from utils.util import AverageMeter
import torch
import matplotlib.pyplot as plt
from models.backbone.helper import load_tf_weights
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from mmcv.runner import load_checkpoint
from utils.prune_utils import BNOptimizer
import torch.nn as nn
import torchvision
from yacscfg import _C as cfg
from yacscfg2 import _C as cfg2
from models.strongerv3 import StrongerV3
from models.strongerv3kl import StrongerV3KL
from models.strongerv3quantile_adjusted import StrongerV3Quantile_Adjusted
from models.strongerv3kl_dropout_alternative import StrongerV3KL_Dropout_Alternative
from models.strongerv3_quantile_dropout import StrongerV3_Quantile_Dropout
from torch import optim
import einops
import math
import numpy as np
class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, args, model, optimizer, lrscheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lrscheduler
        self.experiment_name = args.EXPER.experiment_name
        self.dataset_name = args.DATASET.dataset
        self.dataset_root = args.DATASET.dataset_root

        self.train_dataloader = None
        self.test_dataloader = None
        self.log_iter = self.args.LOG.log_iter

        self.labels = self.args.MODEL.LABEL
        self.num_classes = self.args.MODEL.numcls
        # logger attributes
        self.global_iter = 0
        self.global_epoch = 0
        self.TESTevaluator = None
        self.LossBox = None
        self.LossConf = None
        self.LossClass = None
        self.logger_custom = None
        self.metric_evaluate = None
        self.best_mAP = 0
        self.evaluation_method = ""
        # initialize
        self._get_model()
        self._get_SummaryWriter()
        self._get_dataset()
        self._get_loggers()
        self.sparseBN = []

    def _save_ckpt(self, metric, name=None):
        state = {
            'epoch': self.global_epoch,
            'iter': self.global_iter,
            'state_dict': self.model.state_dict(),
            'opti_dict': self.optimizer.state_dict(),
            'metric': metric
        }
        if name == None:
            torch.save(state, os.path.join(self.save_path, 'checkpoint-{}.pth'.format(self.global_iter)))
        else:
            torch.save(state, os.path.join(self.save_path, 'checkpoint-{}.pth'.format(name)))
        print("save checkpoints at iter{}".format(self.global_iter))

    def _load_ckpt(self):
        if self.args.EXPER.resume == "load_voc":
            load_tf_weights(self.model, 'vocweights.pkl')
        else:  # iter or best
            ckptfile = torch.load(os.path.join(self.save_path, 'checkpoint-{}.pth'.format(self.args.EXPER.resume)))
            state_dict=ckptfile['state_dict']
            new_state_dict= {}
            for k, v in state_dict.items():
                name = k
                #name = "module."+k
                new_state_dict[name]=v
            self.model.load_state_dict(new_state_dict)
            if not self.args.finetune and not self.args.do_test and not self.args.Prune.do_test:
                self.global_epoch = ckptfile['epoch']
                self.global_iter = ckptfile['iter']
                self.best_mAP = ckptfile['metric']
                #self.optimizer.load_state_dict(ckptfile['opti_dict'])
        print("successfully load checkpoint {}".format(self.args.EXPER.resume))
    def _load_ckpt_name(self, name):
        cfg2.merge_from_file("configs/strongerv3_quantile_dropout.yaml")
        net = eval(cfg2.MODEL.modeltype)(cfg=cfg2.MODEL).cuda()
        ckptfile = torch.load(os.path.join('./checkpoints/strongerv3_ensemble/', 'checkpoint-{}.pth'.format(name)))
        state_dict=ckptfile['state_dict']
        new_state_dict= {}
        for k, v in state_dict.items():
            #name=k #remove 'module.' of DataParallel
            name = k[7:]
            new_state_dict[name]=v
        net.load_state_dict(new_state_dict)
        print("successfully load checkpoint {}".format(self.args.EXPER.resume))
        return net;

    def _load_ckpt_quantile_adjusted(self, name):
        cfg2.merge_from_file("configs/strongerv3_kl.yaml")
        net = eval(cfg2.MODEL.modeltype)(cfg=cfg2.MODEL).cuda()
        if self.args.EXPER.resume == "load_voc":
            load_tf_weights(self.model, 'vocweights.pkl')
        else:  # iter or best
            ckptfile = torch.load(os.path.join('./checkpoints/strongerv3_kl/', 'checkpoint-{}.pth'.format(name)))
            state_dict=ckptfile['state_dict']
            new_state_dict= {}
            for k, v in state_dict.items():
                #name=k #remove 'module.' of DataParallel
                name = k
                new_state_dict[name]=v
            net.load_state_dict(new_state_dict)
            backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall = net.get_info()
            self.model.module.load_partial_state(backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall)
            self.optimizer.zero_grad()
    def _load_ckpt_quantile_dropout(self, name):
        cfg2.merge_from_file("configs/strongerv3_kl_dropout_alternative.yaml")
        net = eval(cfg2.MODEL.modeltype)(cfg=cfg2.MODEL).cuda()
        if self.args.EXPER.resume == "load_voc":
            load_tf_weights(self.model, 'vocweights.pkl')
        else:  # iter or best
            if self.experiment_name == "strongerv3_quantile_dropout2":
                ckptfile = torch.load(os.path.join('./checkpoints/gaussian_ensemble_2/', 'checkpoint-{}.pth'.format(name)))
            elif self.experiment_name == "strongerv3_quantile_dropout3":
                ckptfile = torch.load(os.path.join('./checkpoints/gaussian_ensemble_3/', 'checkpoint-{}.pth'.format(name)))
            elif self.experiment_name == "strongerv3_quantile_dropout4":
                ckptfile = torch.load(os.path.join('./checkpoints/gaussian_ensemble_4/', 'checkpoint-{}.pth'.format(name)))
            else:
                ckptfile = torch.load(os.path.join('./checkpoints/gaussian_ensemble_5/', 'checkpoint-{}.pth'.format(name)))
            state_dict=ckptfile['state_dict']
            new_state_dict= {}
            for k, v in state_dict.items():
                #name=k #remove 'module.' of DataParallel
                name = k[7:]
                new_state_dict[name]=v
            net.load_state_dict(new_state_dict)
            backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall = net.get_info()
            self.model.module.load_partial_state(backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall)
            self.optimizer.zero_grad()
    def _load_ckpt_cdf(self, name):
        cfg2.merge_from_file("configs/strongerv3_kl.yaml")
        net = eval(cfg2.MODEL.modeltype)(cfg=cfg2.MODEL).cuda()
        if self.args.EXPER.resume == "load_voc":
            load_tf_weights(self.model, 'vocweights.pkl')
        else:  # iter or best
            ckptfile = torch.load(os.path.join('./checkpoints/strongerv3_kl/', 'checkpoint-{}.pth'.format(name)))
            state_dict=ckptfile['state_dict']
            new_state_dict= {}
            for k, v in state_dict.items():
                #name=k #remove 'module.' of DataParallel
                name = k
                new_state_dict[name]=v
            net.load_state_dict(new_state_dict)
            backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall = net.get_info()
            self.model.module.load_partial_state(backbone,headslarge, mergelarge, headsmid, mergemid, headsmall)
            self.optimizer.zero_grad()
    def _get_model(self):
        self.save_path = './checkpoints/{}/'.format(self.experiment_name)
        ensure_dir(self.save_path)
        self._prepare_device()
        if self.args.EXPER.resume:
            self._load_ckpt()
        elif self.args.EXPER.experiment_name == 'strongerv3_quantile_adjusted':
            self._load_ckpt_quantile_adjusted("best")
        elif self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout2' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout3' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout4':
            self._load_ckpt_quantile_dropout("best")
        elif self.args.EXPER.experiment_name == 'strongerv3_cdf':
            self._load_ckpt_cdf("best")

    def _prepare_device(self):
        if len(self.args.devices)>1:
            self.model=torch.nn.DataParallel(self.model)
    def _get_SummaryWriter(self):
        if not self.args.debug and not self.args.do_test:
            ensure_dir(os.path.join('./summary/', self.experiment_name))

            self.writer = SummaryWriter(log_dir='./summary/{}/{}'.format(self.experiment_name, time.strftime(
                "%m%d-%H-%M-%S", time.localtime(time.time()))))

    def _get_dataset(self):
        self.train_dataloader, self.test_dataloader = eval('get_{}'.format(self.dataset_name))(
            cfg=self.args
        )

    def _get_loggers(self):
        self.LossBox = AverageMeter()
        self.LossConf = AverageMeter()
        self.LossClass = AverageMeter()
        self.logger_losses = {}
        self.logger_losses.update({"lossBox": self.LossBox})
        self.logger_losses.update({"lossConf": self.LossConf})
        self.logger_losses.update({"lossClass": self.LossClass})

    def _reset_loggers(self):
        self.TESTevaluator.reset()
        self.LossClass.reset()
        self.LossConf.reset()
        self.LossBox.reset()

    def updateBN(self):
        for m in self.sparseBN:
            m.weight.grad.data.add_(self.args.Prune.sr * torch.sign(m.weight.data))
    def train(self):
        if self.args.Prune.sparse:
            allbns=[]
            print("start sparse mode")
            for m in self.model.named_modules():
                if isinstance(m[1], nn.BatchNorm2d):
                    allbns.append(m[0])
                    #if 'project_bn' in m[0] or 'dw_bn' in m[0] or 'residual_downsample' in m[0]:
                    if 'dw_bn' in m[0] or 'residual_downsample' in m[0]:
                        continue
                    self.sparseBN.append(m[1])
            print("{}/{} bns will be sparsed.".format(len(self.sparseBN),len(allbns)))
        for epoch in range(self.global_epoch, self.args.OPTIM.total_epoch):
            print("Epoch "+ str(self.global_epoch))
            self.global_epoch += 1

            self._train_epoch()
            self.lr_scheduler.step(epoch)
            lr_current = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("learning_rate", lr_current, epoch)
            for k, v in self.logger_losses.items():
                self.writer.add_scalar(k, v.get_avg(), global_step=self.global_iter)
            self._save_ckpt(metric=0)
            if epoch >= 10:
                results, imgs = self._valid_epoch()
                for k, v in zip(self.logger_custom, results):
                    self.writer.add_scalar(k, v, global_step=self.global_epoch)
                for i in range(len(imgs)):
                    self.writer.add_image("detections_{}".format(i), imgs[i].transpose(2, 0, 1),
                                          global_step=self.global_epoch)
                self._reset_loggers()
                if results[0] > self.best_mAP:
                    self.best_mAP = results[0]
                    self._save_ckpt(name='best', metric=self.best_mAP)

    def _train_epoch(self):
        self.model.train()
        for i, inputs in tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader)):
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            img, _, _, *labels = inputs
            self.global_iter += 1
            if self.global_iter % 100 == 0:
                print(self.global_iter)
                for k, v in self.logger_losses.items():
                    print(k, ":", v.get_avg())
            if self.args.EXPER.experiment_name == 'strongerv3_quantile_adjusted' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout_2' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout_3' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout_4':
                self.train_step_quantile_adjusted(img, labels)
            elif self.args.EXPER.experiment_name == 'strongerv3_cdf':
                self.train_step_cdf(img, labels)
            else:
                self.train_step(img, labels)
                #self.train_step_cdf_gaussian(img, labels)
    def crps_step(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        label_coor = label_sbbox[..., 0:4].view(batch_size,-1,4)
        respond_bbox = label_sbbox[..., 4:5].view(batch_size, -1, 1)
        alpha = torch.rand([self.args.OPTIM.batch_size, 4])
        output = self.model(imgs, alpha)

        #output=output.view(bz,-1,5+self.numclass)

    def train_step(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        outsmall, outmid, outlarge, predsmall, predmid, predlarge = self.model(imgs)
        GIOUloss,conf_loss,probloss = yololoss(self.args.MODEL,outsmall, outmid, outlarge, predsmall, predmid, predlarge,
                             label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
        GIOUloss=GIOUloss.sum()/imgs.shape[0]
        conf_loss=conf_loss.sum()/imgs.shape[0]
        probloss=probloss.sum()/imgs.shape[0]

        totalloss = GIOUloss+conf_loss+probloss
        self.optimizer.zero_grad()
        totalloss.backward()
        if self.args.Prune.sparse:
            self.updateBN()
        self.optimizer.step()
        self.LossBox.update(GIOUloss.item())
        self.LossConf.update(conf_loss.item())
        self.LossClass.update(probloss.item())
    def train_step_quantile_adjusted(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        #print(label_sbbox.view(bz,-1,5+self.numclass))
        alpha = torch.rand([self.args.OPTIM.batch_size, 4])
        outsmall, outmid, outlarge, predsmall, predmid, predlarge = self.model(imgs, alpha)
        GIOUloss,conf_loss,probloss = yololoss_quantile_adjusted(self.args.MODEL,outsmall, outmid, outlarge,
                            predsmall, predmid, predlarge,
                             label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, alpha)
        GIOUloss=GIOUloss.sum()/imgs.shape[0]
        conf_loss=conf_loss.sum()/imgs.shape[0]
        probloss=probloss.sum()/imgs.shape[0]

        totalloss = GIOUloss+conf_loss+probloss
        self.optimizer.zero_grad()
        totalloss.backward()
        if self.args.Prune.sparse:
            self.updateBN()
        self.optimizer.step()
        self.LossBox.update(GIOUloss.item())
        self.LossConf.update(conf_loss.item())
        self.LossClass.update(probloss.item())
    def train_step_cdf(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        slabel_noise = label_sbbox[..., 0:4]+(torch.randn_like(label_sbbox[..., 0:4]))*10
        #print(slabel_noise)
        mlabel_noise = label_mbbox[..., 0:4]+(torch.randn_like(label_mbbox[..., 0:4]))*10
        llabel_noise = label_lbbox[..., 0:4]+(torch.randn_like(label_lbbox[..., 0:4]))*10

        slabel = slabel_noise.reshape(label_sbbox.shape[0], label_sbbox.shape[1], label_sbbox.shape[2], label_sbbox.shape[3]*4).permute(0, 3, 1, 2)
        mlabel = mlabel_noise.reshape(label_mbbox.shape[0], label_mbbox.shape[1], label_mbbox.shape[2], label_mbbox.shape[3]*4).permute(0, 3, 1, 2)
        llabel = llabel_noise.reshape(label_lbbox.shape[0], label_lbbox.shape[1], label_lbbox.shape[2], label_lbbox.shape[3]*4).permute(0, 3, 1, 2)
        outsmall, outmid, outlarge, predsmall, predmid, predlarge = self.model(imgs, llabel, mlabel, slabel)
        #print(predlarge[...,0:4])
        GIOUloss,conf_loss,probloss = yololoss_cdf(self.args.MODEL,outsmall, outmid, outlarge,
                            predsmall, predmid, predlarge,
                             label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, llabel_noise, mlabel_noise, slabel_noise)
        GIOUloss=GIOUloss.sum()/imgs.shape[0]
        conf_loss=conf_loss.sum()/imgs.shape[0]
        probloss=probloss.sum()/imgs.shape[0]

        totalloss = GIOUloss+conf_loss+probloss
        self.optimizer.zero_grad()
        totalloss.backward()
        if self.args.Prune.sparse:
            self.updateBN()
        self.optimizer.step()
        self.LossBox.update(GIOUloss.item())
        self.LossConf.update(conf_loss.item())
        self.LossClass.update(probloss.item())
        # for k, v in self.logger_losses.items():
        #     print(k, ":", v.get_avg())
    def train_step_cdf_gaussian(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        slabel_noise = label_sbbox[..., 0:4]+(torch.rand_like(label_sbbox[..., 0:4]))*10
        #print(slabel_noise)
        mlabel_noise = label_mbbox[..., 0:4]+(torch.rand_like(label_mbbox[..., 0:4]))*10
        llabel_noise = label_lbbox[..., 0:4]+(torch.rand_like(label_lbbox[..., 0:4]))*10

        slabel = slabel_noise.reshape(label_sbbox.shape[0], label_sbbox.shape[1], label_sbbox.shape[2], label_sbbox.shape[3]*4).permute(0, 3, 1, 2)
        mlabel = mlabel_noise.reshape(label_mbbox.shape[0], label_mbbox.shape[1], label_mbbox.shape[2], label_mbbox.shape[3]*4).permute(0, 3, 1, 2)
        llabel = llabel_noise.reshape(label_lbbox.shape[0], label_lbbox.shape[1], label_lbbox.shape[2], label_lbbox.shape[3]*4).permute(0, 3, 1, 2)
        outsmall, outmid, outlarge, predsmall, predmid, predlarge = self.model(imgs)
        #print(predlarge[...,0:4])
        GIOUloss = yololoss_cdf_gaussian(self.args.MODEL,outsmall, outmid, outlarge,
                            predsmall, predmid, predlarge,
                             label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, llabel_noise, mlabel_noise, slabel_noise)
        GIOUloss=GIOUloss.sum()/imgs.shape[0]
        self.LossBox.update(GIOUloss.item())
    def sample_cdf(self):
        self.model.eval()
        for i, inputs in tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader)):
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            imgs, _, ori_shapes, *labels = inputs
            labels = [label.cuda() for label in labels]
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
            for i in range(20):
                slabel_sample = torch.rand((label_sbbox[..., 0:4].shape[0],label_sbbox[..., 0:4].shape[1],label_sbbox[..., 0:4].shape[2],label_sbbox[..., 0:4].shape[3], label_sbbox[..., 0:4].shape[4]))
                mlabel_sample = torch.rand((label_mbbox[..., 0:4].shape[0],label_mbbox[..., 0:4].shape[1],label_mbbox[..., 0:4].shape[2],label_mbbox[..., 0:4].shape[3], label_sbbox[..., 0:4].shape[4]))
                llabel_sample = torch.rand((label_lbbox[..., 0:4].shape[0],label_lbbox[..., 0:4].shape[1],label_lbbox[..., 0:4].shape[2],label_lbbox[..., 0:4].shape[3], label_sbbox[..., 0:4].shape[4]))
                ori_shapes = ori_shapes.cuda()
                for imgidx in range(imgs.shape[0]):
                    org_h, org_w = ori_shapes[imgidx]
                    slabel_sample[imgidx] = slabel_sample[imgidx]*torch.tensor([org_h, org_w, org_h, org_w])
                    mlabel_sample[imgidx] = mlabel_sample[imgidx]*torch.tensor([org_h, org_w, org_h, org_w])
                    llabel_sample[imgidx] = llabel_sample[imgidx]*torch.tensor([org_h, org_w, org_h, org_w])
                slabel = slabel_sample.reshape(label_sbbox.shape[0], label_sbbox.shape[1], label_sbbox.shape[2], label_sbbox.shape[3]*4).permute(0, 3, 1, 2)
                mlabel = mlabel_sample.reshape(label_mbbox.shape[0], label_mbbox.shape[1], label_mbbox.shape[2], label_mbbox.shape[3]*4).permute(0, 3, 1, 2)
                llabel = llabel_sample.reshape(label_lbbox.shape[0], label_lbbox.shape[1], label_lbbox.shape[2], label_lbbox.shape[3]*4).permute(0, 3, 1, 2)
                # slabel_noise = label_sbbox[..., 0:4]+(torch.randn_like(label_sbbox[..., 0:4]) - 0.5)*3
                # mlabel_noise = label_mbbox[..., 0:4]+(torch.randn_like(label_mbbox[..., 0:4]) - 0.5)*3
                # llabel_noise = label_lbbox[..., 0:4]+(torch.randn_like(label_lbbox[..., 0:4]) - 0.5)*3
                #
                # slabel = slabel_noise.reshape(label_sbbox.shape[0], label_sbbox.shape[1], label_sbbox.shape[2], label_sbbox.shape[3]*4).permute(0, 3, 1, 2)
                # mlabel = mlabel_noise.reshape(label_mbbox.shape[0], label_mbbox.shape[1], label_mbbox.shape[2], label_mbbox.shape[3]*4).permute(0, 3, 1, 2)
                # llabel = llabel_noise.reshape(label_lbbox.shape[0], label_lbbox.shape[1], label_lbbox.shape[2], label_lbbox.shape[3]*4).permute(0, 3, 1, 2)
                outputs = self.model(imgs, llabel, mlabel, slabel)
            print(outputs[:, :, 0:4])
    def sample_quantile_load(self, numsamples, validiter=-1):
        self.TESTevaluator.load_dict()
        # self.TESTevaluator.get_calibration_samples()
        # self.TESTevaluator.get_distribution_samples()
        results = self.TESTevaluator.evaluate()
        imgs = self.TESTevaluator.visual_imgs
        for k, v in zip(self.logger_custom, results):
            print("{}:{}".format(k, v))
        return None
    def sample_quantile(self, num_samples,validiter=-1):
      def _postprocess(pred_bbox, test_input_size, org_img_shape):
          if self.args.MODEL.boxloss == 'KL' or self.args.MODEL.boxloss == 'quantile':
              pred_coor = pred_bbox[:, 0:4]
              pred_vari = pred_bbox[:, 4:8]
              pred_vari = torch.exp(pred_vari)
              pred_conf = pred_bbox[:, 8]
              pred_prob = pred_bbox[:, 9:]
          else:
              pred_coor = pred_bbox[:, 0:4]
              pred_conf = pred_bbox[:, 4]
              pred_prob = pred_bbox[:, 5:]
          org_h, org_w = org_img_shape
          resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
          dw = (test_input_size - resize_ratio * org_w) / 2
          dh = (test_input_size - resize_ratio * org_h) / 2
          pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
          pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
          x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
          x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
          x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
          pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)
          # ***********************
          if pred_prob.shape[-1]==0:
              pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
          # ***********************
          scores = pred_conf.unsqueeze(-1) * pred_prob
          bboxes = torch.cat([pred_coor, scores], dim=-1)
          return bboxes,pred_vari
      s = time.time()
      self.model.eval()
      #return self.sample_quantile_load(10)
      for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
      #for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
          if idx_batch == validiter:  # to save time
              break
          inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
          (imgs, imgpath, ori_shapes, *_) = inputs
          imgs = imgs.cuda()
          ori_shapes = ori_shapes.cuda()
          with torch.no_grad():
              convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
          with torch.no_grad():
              outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)

          bbox_array = [torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)]
          for i in range(num_samples):
              #alpha = torch.cat([torch.full([self.args.OPTIM.batch_size, 2], fill_value = 0.1+0.8*i/(num_samples-1)), torch.full([self.args.OPTIM.batch_size, 2], fill_value = 0.9-0.8*i/(num_samples-1))], dim = 1)
              alpha =  torch.rand([self.args.OPTIM.batch_size, 4])##torch.rand([self.args.OPTIM.batch_size, 4])
              with torch.no_grad():
                  outputs = self.model.module.partial_forward_2(convlarge, convmid, convsmall, outlarge_orig, outmid_orig, outsmall_orig, alpha)
                  bbox_array.append(outputs)
          bbox_final = torch.stack(bbox_array, dim = 3)
          for imgidx in range(len(outputs)):
              postprocessed_array = []
              for i in range(num_samples+1):
                  bbox, bboxvari = _postprocess(bbox_final[imgidx, :, :, i], imgs.shape[-1], ori_shapes[imgidx])
                  postprocessed_array.append(bbox)
              bbox = torch.stack(postprocessed_array, dim = 2)
              nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                               variance=bboxvari)
              if nms_boxes is not None:
                  self.TESTevaluator.append(imgpath[imgidx][0],
                                            nms_boxes.cpu().numpy(),
                                            nms_scores.cpu().numpy(),
                                            nms_labels.cpu().numpy(),
                                            bboxvari.cpu().numpy())

      self.TESTevaluator.save_dict()
      self.TESTevaluator.load_dict()
      self.TESTevaluator.get_distribution_samples()
      results = self.TESTevaluator.evaluate()
      imgs = self.TESTevaluator.visual_imgs
      for k, v in zip(self.logger_custom, results):
          print("{}:{}".format(k, v))
      return results, imgs
    def sample_pure_variational(self, num_samples,validiter=-1):
      def _postprocess(pred_bbox, test_input_size, org_img_shape):
          if self.args.MODEL.boxloss == 'KL' or self.args.MODEL.boxloss == 'quantile':
              pred_coor = pred_bbox[:, 0:4]
              pred_vari = pred_bbox[:, 4:8]
              pred_vari = torch.exp(pred_vari)
              pred_conf = pred_bbox[:, 8]
              pred_prob = pred_bbox[:, 9:]
          else:
              pred_coor = pred_bbox[:, 0:4]
              pred_conf = pred_bbox[:, 4]
              pred_prob = pred_bbox[:, 5:]
          org_h, org_w = org_img_shape
          resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
          dw = (test_input_size - resize_ratio * org_w) / 2
          dh = (test_input_size - resize_ratio * org_h) / 2
          pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
          pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
          x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
          x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
          x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
          pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)
          # ***********************
          if pred_prob.shape[-1]==0:
              pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
          # ***********************
          scores = pred_conf.unsqueeze(-1) * pred_prob
          bboxes = torch.cat([pred_coor, scores], dim=-1)
          return bboxes,pred_vari
      s = time.time()
      self.model.train()
      #return self.sample_quantile_load(10)
      for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
      #for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
          if idx_batch == validiter:  # to save time
              break
          inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
          (imgs, imgpath, ori_shapes, *_) = inputs
          imgs = imgs.cuda()
          ori_shapes = ori_shapes.cuda()
          outlarge_array = []
          outmid_array = []
          outsmall_array = []

          self.model.eval()
          with torch.no_grad():
              convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
              outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
          bbox_array = [torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)]
          self.model.train()
          for i in range(200):
              with torch.no_grad():
                  outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                  outputs = torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)
                  bbox_array.append(outputs)
          bbox_final = torch.stack(bbox_array, dim = 3)
          for imgidx in range(len(outputs)):
              postprocessed_array = []
              for i in range(bbox_final.size()[3]):
                  bbox, bboxvari = _postprocess(bbox_final[imgidx, :, :, i], imgs.shape[-1], ori_shapes[imgidx])
                  postprocessed_array.append(bbox)
              bbox = torch.stack(postprocessed_array, dim = 2)
              nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                               variance=bboxvari)
              if nms_boxes is not None:
                  self.TESTevaluator.append(imgpath[imgidx][0],
                                            nms_boxes.cpu().numpy(),
                                            nms_scores.cpu().numpy(),
                                            nms_labels.cpu().numpy(),
                                            bboxvari.cpu().numpy())

      self.TESTevaluator.save_dict()
      self.TESTevaluator.load_dict()
      self.TESTevaluator.get_distribution_samples()
      results = self.TESTevaluator.evaluate()
      imgs = self.TESTevaluator.visual_imgs
      for k, v in zip(self.logger_custom, results):
          print("{}:{}".format(k, v))
      return results, imgs
    def sample_quantile_variational(self, num_samples,validiter=-1):
      def _postprocess(pred_bbox, test_input_size, org_img_shape):
          if self.args.MODEL.boxloss == 'KL' or self.args.MODEL.boxloss == 'quantile':
              pred_coor = pred_bbox[:, 0:4]
              pred_vari = pred_bbox[:, 4:8]
              pred_vari = torch.exp(pred_vari)
              pred_conf = pred_bbox[:, 8]
              pred_prob = pred_bbox[:, 9:]
          else:
              pred_coor = pred_bbox[:, 0:4]
              pred_conf = pred_bbox[:, 4]
              pred_prob = pred_bbox[:, 5:]
          org_h, org_w = org_img_shape
          resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
          dw = (test_input_size - resize_ratio * org_w) / 2
          dh = (test_input_size - resize_ratio * org_h) / 2
          pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
          pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
          x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
          x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
          x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
          pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)
          # ***********************
          if pred_prob.shape[-1]==0:
              pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
          # ***********************
          scores = pred_conf.unsqueeze(-1) * pred_prob
          bboxes = torch.cat([pred_coor, scores], dim=-1)
          return bboxes,pred_vari
      s = time.time()
      self.model.train()
      #return self.sample_quantile_load(10)
      for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
      #for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
          if idx_batch == validiter:  # to save time
              break
          inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
          (imgs, imgpath, ori_shapes, *_) = inputs
          imgs = imgs.cuda()
          ori_shapes = ori_shapes.cuda()
          outlarge_array = []
          outmid_array = []
          outsmall_array = []

          self.model.eval()
          with torch.no_grad():
              convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
              outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
          bbox_array = [torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)]
          self.model.train()
          for i in range(10):
              with torch.no_grad():
                  outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
          #         outlarge_array.append(outlarge_orig)
          #         outmid_array.append(outmid_orig)
          #         outsmall_array.append(outsmall_orig)
          # outlarge_orig = torch.stack(outlarge_array, dim = 4).mean(dim = 4)
          # outmid_orig = torch.stack(outmid_array, dim = 4).mean(dim = 4)
          # outsmall_orig = torch.stack(outsmall_array, dim = 4).mean(dim = 4)
          #bbox_array = [torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)]
              for i in range(num_samples):
                  alpha =  torch.rand([self.args.OPTIM.batch_size, 4])##torch.rand([self.args.OPTIM.batch_size, 4])
                  with torch.no_grad():
                      outputs = self.model.module.partial_forward_2(convlarge, convmid, convsmall, outlarge_orig, outmid_orig, outsmall_orig, alpha)
                      bbox_array.append(outputs)
          bbox_final = torch.stack(bbox_array, dim = 3)
          for imgidx in range(len(outputs)):
              postprocessed_array = []
              for i in range(bbox_final.size()[3]):
                  bbox, bboxvari = _postprocess(bbox_final[imgidx, :, :, i], imgs.shape[-1], ori_shapes[imgidx])
                  postprocessed_array.append(bbox)
              bbox = torch.stack(postprocessed_array, dim = 2)
              nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                               variance=bboxvari)
              if nms_boxes is not None:
                  self.TESTevaluator.append(imgpath[imgidx][0],
                                            nms_boxes.cpu().numpy(),
                                            nms_scores.cpu().numpy(),
                                            nms_labels.cpu().numpy(),
                                            bboxvari.cpu().numpy())

      self.TESTevaluator.save_dict()
      self.TESTevaluator.load_dict()
      self.TESTevaluator.get_distribution_samples()
      results = self.TESTevaluator.evaluate()
      imgs = self.TESTevaluator.visual_imgs
      for k, v in zip(self.logger_custom, results):
          print("{}:{}".format(k, v))
      return results, imgs
    def sample_quantile_ensemble(self, num_samples,validiter=-1):
      def _postprocess(pred_bbox, test_input_size, org_img_shape):
          if self.args.MODEL.boxloss == 'KL' or self.args.MODEL.boxloss == 'quantile':
              pred_coor = pred_bbox[:, 0:4]
              pred_vari = pred_bbox[:, 4:8]
              pred_vari = torch.exp(pred_vari)
              pred_conf = pred_bbox[:, 8]
              pred_prob = pred_bbox[:, 9:]
          else:
              pred_coor = pred_bbox[:, 0:4]
              pred_conf = pred_bbox[:, 4]
              pred_prob = pred_bbox[:, 5:]
          org_h, org_w = org_img_shape
          resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
          dw = (test_input_size - resize_ratio * org_w) / 2
          dh = (test_input_size - resize_ratio * org_h) / 2
          pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
          pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
          x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
          x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
          x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
          pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)
          # ***********************
          if pred_prob.shape[-1]==0:
              pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
          # ***********************
          scores = pred_conf.unsqueeze(-1) * pred_prob
          bboxes = torch.cat([pred_coor, scores], dim=-1)
          return bboxes,pred_vari
      s = time.time()
      model2 = self._load_ckpt_name("dropout2")
      model3 = self._load_ckpt_name("dropout3")
      #model4 = self._load_ckpt_name("dropout4")
      modellist = [self.model.module, model2, model3]
      for model in modellist:
          model.train()
      #model4.eval()
      #return self.sample_quantile_load(10)
      for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
      #for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
          if idx_batch == validiter:  # to save time
              break
          inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
          (imgs, imgpath, ori_shapes, *_) = inputs
          imgs = imgs.cuda()
          ori_shapes = ori_shapes.cuda()
          bbox_final_array = []
          for model in modellist:
              outlarge_array = []
              outmid_array = []
              outsmall_array = []
              with torch.no_grad():
                  convlarge, convmid, convsmall = model.partial_forward(imgs)
              for i in range(num_samples):
                  with torch.no_grad():
                      outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                      outlarge_array.append(outlarge_orig)
                      outmid_array.append(outmid_orig)
                      outsmall_array.append(outsmall_orig)
              outlarge_orig = torch.stack(outlarge_array, dim = 4).mean(dim = 4)
              outmid_orig = torch.stack(outmid_array, dim = 4).mean(dim = 4)
              outsmall_orig = torch.stack(outsmall_array, dim = 4).mean(dim = 4)
              bbox_array = [torch.cat([model.decode_infer(outsmall_orig, 8), model.decode_infer(outmid_orig, 16), model.decode_infer(outlarge_orig, 32)], dim = 1)]
              for i in range(num_samples):
                  alpha =  torch.rand([self.args.OPTIM.batch_size, 4])##torch.rand([self.args.OPTIM.batch_size, 4])
                  with torch.no_grad():
                      outputs = model.partial_forward_2(convlarge, convmid, convsmall, outlarge_orig, outmid_orig, outsmall_orig, alpha)
                      bbox_array.append(outputs)
                  bbox_final = torch.stack(bbox_array, dim = 3)
              bbox_final_array.append(bbox_final)
          bbox_final = torch.stack(bbox_final_array, dim = 4).mean(dim = 4)
          for imgidx in range(len(outputs)):
              postprocessed_array = []
              for i in range(num_samples+1):
                  bbox, bboxvari = _postprocess(bbox_final[imgidx, :, :, i], imgs.shape[-1], ori_shapes[imgidx])
                  postprocessed_array.append(bbox)
              bbox = torch.stack(postprocessed_array, dim = 2)
              nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                               variance=bboxvari)
              if nms_boxes is not None:
                  self.TESTevaluator.append(imgpath[imgidx][0],
                                            nms_boxes.cpu().numpy(),
                                            nms_scores.cpu().numpy(),
                                            nms_labels.cpu().numpy(),
                                            bboxvari.cpu().numpy())

      self.TESTevaluator.save_dict()
      self.TESTevaluator.load_dict()
      self.TESTevaluator.get_distribution_samples()
      results = self.TESTevaluator.evaluate()
      imgs = self.TESTevaluator.visual_imgs
      for k, v in zip(self.logger_custom, results):
          print("{}:{}".format(k, v))
      return results, imgs
    def sample_gaussian_ensemble(self, num_samples,validiter=-1):
      def _postprocess(pred_bbox, test_input_size, org_img_shape):
          if self.args.MODEL.boxloss == 'KL' or self.args.MODEL.boxloss == 'quantile':
              pred_coor = pred_bbox[:, 0:4]
              pred_vari = pred_bbox[:, 4:8]
              pred_vari = torch.exp(pred_vari)
              pred_conf = pred_bbox[:, 8]
              pred_prob = pred_bbox[:, 9:]
          else:
              pred_coor = pred_bbox[:, 0:4]
              pred_conf = pred_bbox[:, 4]
              pred_prob = pred_bbox[:, 5:]
          org_h, org_w = org_img_shape
          resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
          dw = (test_input_size - resize_ratio * org_w) / 2
          dh = (test_input_size - resize_ratio * org_h) / 2
          pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
          pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
          x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
          x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
          x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
          pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)
          # ***********************
          if pred_prob.shape[-1]==0:
              pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
          # ***********************
          scores = pred_conf.unsqueeze(-1) * pred_prob
          bboxes = torch.cat([pred_coor, scores], dim=-1)
          return bboxes,pred_vari
      s = time.time()
      model2 = self._load_ckpt_name("dropout2")
      model3 = self._load_ckpt_name("dropout3")
      #model4 = self._load_ckpt_name("dropout4")
      modellist = [self.model.module, model2, model3]
      for model in modellist:
          model.eval()
      #model4.eval()
      #return self.sample_quantile_load(10)
      for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
      #for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
          if idx_batch == validiter:  # to save time
              break
          inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
          (imgs, imgpath, ori_shapes, *_) = inputs
          imgs = imgs.cuda()
          ori_shapes = ori_shapes.cuda()
          output_array = []
          for model in modellist:
              outlarge_array = []
              outmid_array = []
              outsmall_array = []
              with torch.no_grad():
                  convlarge, convmid, convsmall = model.partial_forward(imgs)
              for i in range(num_samples):
                  with torch.no_grad():
                      outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                      outlarge_array.append(outlarge_orig)
                      outmid_array.append(outmid_orig)
                      outsmall_array.append(outsmall_orig)
              outlarge_orig = torch.stack(outlarge_array, dim = 4).mean(dim = 4)
              outmid_orig = torch.stack(outmid_array, dim = 4).mean(dim = 4)
              outsmall_orig = torch.stack(outsmall_array, dim = 4).mean(dim = 4)
              outputs = torch.cat([model.decode_infer(outsmall_orig, 8), model.decode_infer(outmid_orig, 16), model.decode_infer(outlarge_orig, 32)], dim = 1)
              output_array.append(outputs)
          outputs = torch.stack(output_array, dim = 2).mean(dim = 2)
          for imgidx in range(len(outputs)):
              bbox,bboxvari = _postprocess(outputs[imgidx], imgs.shape[-1], ori_shapes[imgidx])
              #print(bboxvari)
              #bbox2 = einops.repeat(bbox, 'm n -> m n k', k=(num_samples*19+1))
              bbox = einops.repeat(bbox, 'm n -> m n k', k=(num_samples+1))

              #bbox[:, 0:4, 1:] = torch.sort(torch.normal(bbox2[:, 0:4, :], einops.repeat(torch.sqrt(bboxvari), 'm n -> m n k', k=num_samples*19+1)), dim = -1, descending = False)[0][:, :, ::20]
              bbox[:, 0:4, 1:] = torch.normal(bbox[:, 0:4, 1:], einops.repeat(torch.sqrt(bboxvari), 'm n -> m n k', k=num_samples))
              nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                               variance=bboxvari)
              if nms_boxes is not None:
                  self.TESTevaluator.append(imgpath[imgidx][0],
                                            nms_boxes.cpu().numpy(),
                                            nms_scores.cpu().numpy(),
                                            nms_labels.cpu().numpy(),
                                            bboxvari.cpu().numpy())

      self.TESTevaluator.save_dict()
      self.TESTevaluator.load_dict()
      self.TESTevaluator.get_distribution_samples()
      results = self.TESTevaluator.evaluate()
      imgs = self.TESTevaluator.visual_imgs
      for k, v in zip(self.logger_custom, results):
          print("{}:{}".format(k, v))
      return results, imgs
    def sample_gaussian(self,num_samples, validiter=-1):
        def _postprocess(pred_bbox, test_input_size, org_img_shape):
            pred_coor = pred_bbox[:, 0:4]
            pred_vari = pred_bbox[:, 4:8]
            pred_vari = torch.exp(pred_vari)
            pred_conf = pred_bbox[:, 8]
            pred_prob = pred_bbox[:, 9:]
            org_h, org_w = org_img_shape
            resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
            dw = (test_input_size - resize_ratio * org_w) / 2
            dh = (test_input_size - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
            x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
            x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
            x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
            pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)

            # ***********************
            if pred_prob.shape[-1]==0:
                pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
            # ***********************
            scores = pred_conf.unsqueeze(-1) * pred_prob
            bboxes = torch.cat([pred_coor, scores], dim=-1)
            return bboxes,pred_vari
        s = time.time()
        self.model.eval()
        #for idx_batch, inputs in tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader)):
        for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
            torch.cuda.empty_cache()
            if idx_batch == validiter:  # to save time
                break
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            (imgs, imgpath, ori_shapes, *_) = inputs
            imgs = imgs.cuda()
            ori_shapes = ori_shapes.cuda()
            with torch.no_grad():
                #outputs = self.model(imgs)
                convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
                outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                outputs = torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)
                #outputs = self.model(imgs, alpha)
                #print(outputs)
            for imgidx in range(len(outputs)):
                bbox,bboxvari = _postprocess(outputs[imgidx], imgs.shape[-1], ori_shapes[imgidx])
                #print(bboxvari)
                #bbox2 = einops.repeat(bbox, 'm n -> m n k', k=(num_samples*19+1))
                bbox = einops.repeat(bbox, 'm n -> m n k', k=(num_samples+1))

                #bbox[:, 0:4, 1:] = torch.sort(torch.normal(bbox2[:, 0:4, :], einops.repeat(torch.sqrt(bboxvari), 'm n -> m n k', k=num_samples*19+1)), dim = -1, descending = False)[0][:, :, ::20]
                bbox[:, 0:4, 1:] = torch.normal(bbox[:, 0:4, 1:], einops.repeat(torch.sqrt(bboxvari), 'm n -> m n k', k=num_samples))
                nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                                 variance=bboxvari)
                if nms_boxes is not None:
                    self.TESTevaluator.append(imgpath[imgidx][0],
                                              nms_boxes.cpu().numpy(),
                                              nms_scores.cpu().numpy(),
                                              nms_labels.cpu().numpy(),
                                              bboxvari.cpu().numpy())
        self.TESTevaluator.save_dict()
        self.TESTevaluator.load_dict()
        self.TESTevaluator.get_distribution_samples()
        results = self.TESTevaluator.evaluate()
        imgs = self.TESTevaluator.visual_imgs
        for k, v in zip(self.logger_custom, results):
            print("{}:{}".format(k, v))
        return results, imgs
    def sample_gaussian_variational(self,num_samples, validiter=-1):
        def _postprocess(pred_bbox, test_input_size, org_img_shape):
            pred_coor = pred_bbox[:, 0:4]
            pred_vari = pred_bbox[:, 4:8]
            pred_vari = torch.exp(pred_vari)
            pred_conf = pred_bbox[:, 8]
            pred_prob = pred_bbox[:, 9:]
            org_h, org_w = org_img_shape
            resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
            dw = (test_input_size - resize_ratio * org_w) / 2
            dh = (test_input_size - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
            x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
            x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
            x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
            pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)

            # ***********************
            if pred_prob.shape[-1]==0:
                pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
            # ***********************
            scores = pred_conf.unsqueeze(-1) * pred_prob
            bboxes = torch.cat([pred_coor, scores], dim=-1)
            return bboxes,pred_vari
        s = time.time()
        self.model.train()
        #for idx_batch, inputs in tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader)):
        for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
            torch.cuda.empty_cache()
            if idx_batch == validiter:  # to save time
                break
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            (imgs, imgpath, ori_shapes, *_) = inputs
            imgs = imgs.cuda()
            ori_shapes = ori_shapes.cuda()
            with torch.no_grad():
                convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
                outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                outlarge_array = []
                outmid_array = []
                outsmall_array = []

                with torch.no_grad():
                    convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
                for i in range(10):
                    with torch.no_grad():
                        outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                        outlarge_array.append(outlarge_orig)
                        outmid_array.append(outmid_orig)
                        outsmall_array.append(outsmall_orig)
                self.model.eval()
                with torch.no_grad():
                    outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                outputs_orig = torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)
                # outlarge_orig = torch.stack(outlarge_array, dim = 4).mean(dim = 4)
                # outmid_orig = torch.stack(outmid_array, dim = 4).mean(dim = 4)
                # outsmall_orig = torch.stack(outsmall_array, dim = 4).mean(dim = 4)
                # outputs = torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)
                #outputs = self.model(imgs, alpha)
                #print(outputs)
            for imgidx in range(len(outputs_orig)):
                bbox_array = []
                for i in range(10):
                    outputs = torch.cat([self.model.module.decode_infer(outsmall_array[i], 8), self.model.module.decode_infer(outmid_array[i], 16), self.model.module.decode_infer(outlarge_array[i], 32)], dim = 1)
                    bbox,bboxvari = _postprocess(outputs[imgidx], imgs.shape[-1], ori_shapes[imgidx])
                    bbox = einops.repeat(bbox, 'm n -> m n k', k=(num_samples))

                    bbox[:, 0:4, :] = torch.normal(bbox[:, 0:4, :], einops.repeat(torch.sqrt(bboxvari), 'm n -> m n k', k=num_samples))
                    bbox_array.append(bbox)
                bbox = torch.cat(bbox_array, dim = 2)
                bbox_orig,bboxvari = _postprocess(outputs_orig[imgidx], imgs.shape[-1], ori_shapes[imgidx])
                bbox = torch.cat((bbox_orig.unsqueeze(2), bbox), dim = 2)
                nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                                     variance=bboxvari)
                if nms_boxes is not None:
                    self.TESTevaluator.append(imgpath[imgidx][0],
                                              nms_boxes.cpu().numpy(),
                                              nms_scores.cpu().numpy(),
                                              nms_labels.cpu().numpy(),
                                              bboxvari.cpu().numpy())
        self.TESTevaluator.save_dict()
        self.TESTevaluator.load_dict()
        self.TESTevaluator.get_distribution_samples()
        results = self.TESTevaluator.evaluate()
        imgs = self.TESTevaluator.visual_imgs
        for k, v in zip(self.logger_custom, results):
            print("{}:{}".format(k, v))
        return results, imgs
    def sample_mdn(self,num_samples, validiter=-1):
        def _postprocess(pred_bbox, test_input_size, org_img_shape):
            pred_coor = pred_bbox[:, 0:4]
            pred_vari = pred_bbox[:, 4:8]
            pred_vari = torch.exp(pred_vari)
            pred_conf = pred_bbox[:, 8]
            pred_prob = pred_bbox[:, 9:]
            org_h, org_w = org_img_shape
            resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
            dw = (test_input_size - resize_ratio * org_w) / 2
            dh = (test_input_size - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
            x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
            x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
            x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
            pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)

            # ***********************
            if pred_prob.shape[-1]==0:
                pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
            # ***********************
            scores = pred_conf.unsqueeze(-1) * pred_prob
            bboxes = torch.cat([pred_coor, scores], dim=-1)
            return bboxes,pred_vari
        s = time.time()
        self.model.eval()
        #for idx_batch, inputs in tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader)):
        for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
            torch.cuda.empty_cache()
            if idx_batch == validiter:  # to save time
                break
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            (imgs, imgpath, ori_shapes, *_) = inputs
            imgs = imgs.cuda()
            ori_shapes = ori_shapes.cuda()
            with torch.no_grad():
                outputs = self.model(imgs)

            for imgidx in range(len(outputs)):
                bbox,bboxvari = _postprocess(outputs[imgidx], imgs.shape[-1], ori_shapes[imgidx])
                #print(bboxvari)
                bbox = einops.repeat(bbox, 'm n -> m n k', k=(num_samples+1))
                bbox[:, 0:4, 1:] = torch.normal(bbox[:, 0:4, 1:], einops.repeat(torch.sqrt(bboxvari), 'm n -> m n k', k=num_samples))
                nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
                                                                 variance=bboxvari)
                if nms_boxes is not None:
                    self.TESTevaluator.append(imgpath[imgidx][0],
                                              nms_boxes.cpu().numpy(),
                                              nms_scores.cpu().numpy(),
                                              nms_labels.cpu().numpy(),
                                              bboxvari.cpu().numpy())
        self.TESTevaluator.save_dict()
        self.TESTevaluator.load_dict()
        self.TESTevaluator.get_calibration_samples()
        self.TESTevaluator.get_distribution_samples()
        results = self.TESTevaluator.evaluate()
        imgs = self.TESTevaluator.visual_imgs
        for k, v in zip(self.logger_custom, results):
            print("{}:{}".format(k, v))
        return results, imgs
    def transformAlpha(self, alpha):
        return np.tan((alpha)*math.pi*0.95-0.95*math.pi/2)
    def _valid_epoch(self,validiter=-1):
        def _postprocess(pred_bbox, test_input_size, org_img_shape):
            if self.args.MODEL.boxloss == 'KL' or self.args.MODEL.boxloss == 'quantile':
                pred_coor = pred_bbox[:, 0:4]
                pred_vari = pred_bbox[:, 4:8]
                pred_vari = torch.exp(pred_vari)
                pred_conf = pred_bbox[:, 8]
                pred_prob = pred_bbox[:, 9:]
            else:
                pred_coor = pred_bbox[:, 0:4]
                pred_conf = pred_bbox[:, 4]
                pred_prob = pred_bbox[:, 5:]
            org_h, org_w = org_img_shape
            resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
            dw = (test_input_size - resize_ratio * org_w) / 2
            dh = (test_input_size - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
            x1,y1,x2,y2=torch.split(pred_coor,[1,1,1,1],dim=1)
            x1,y1=torch.max(x1,torch.zeros_like(x1)),torch.max(y1,torch.zeros_like(y1))
            x2,y2=torch.min(x2,torch.ones_like(x2)*(org_w-1)),torch.min(y2,torch.ones_like(y2)*(org_h-1))
            pred_coor=torch.cat([x1,y1,x2,y2],dim=-1)

            # ***********************
            if pred_prob.shape[-1]==0:
                pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
            # ***********************
            scores = pred_conf.unsqueeze(-1) * pred_prob
            bboxes = torch.cat([pred_coor, scores], dim=-1)
            if self.args.MODEL.boxloss == 'KL' and self.args.EVAL.varvote:
                return bboxes, pred_vari
            else:
                return bboxes,None
        if self.args.EXPER.experiment_name == 'strongerv3_cdf':
            return self.sample_cdf()

        evaluation_method = "crps"
        if evaluation_method == "crps":
            return self.sample_pure_variational(20, validiter=-1)
        s = time.time()
        self.model.eval()
        for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
            if idx_batch == validiter:  # to save time
                break
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            (imgs, imgpath, ori_shapes, *_) = inputs
            imgs = imgs.cuda()
            ori_shapes = ori_shapes.cuda()
            with torch.no_grad():
                if self.args.EXPER.experiment_name == "strongerv3_quantile" or self.args.EXPER.experiment_name == "strongerv3_quantile_adjusted" or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout_2' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout_3' or self.args.EXPER.experiment_name == 'strongerv3_quantile_dropout_4':
                     alpha = torch.full([self.args.OPTIM.batch_size, 4], fill_value = 0.5)
                     outputs = self.model(imgs, alpha)
                     convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
                     outlarge_orig, outmid_orig, outsmall_orig = self.model.module.partial_forward_orig(convlarge, convmid, convsmall)
                     outputs = torch.cat([self.model.module.decode_infer(outsmall_orig, 8), self.model.module.decode_infer(outmid_orig, 16), self.model.module.decode_infer(outlarge_orig, 32)], dim = 1)
                elif self.args.EXPER.experiment_name == "strongerv3_mdn":
                    outputs = self.model(imgs)
                    pred_pi = outputs[..., 0:20]
                    pred_coor = outputs[..., 20:40]
                    pred_vari = outputs[..., 40:60]
                    pred_rest = outputs[..., 60:]
                    box_sides = []
                    for i in range(4):
                        box_sides.append(self.mean(pred_pi[..., i*5:i*5+5].view(-1, 5), torch.exp(-pred_vari[..., i*5:i*5+5]).view(-1, 5), pred_coor[..., i*5:i*5+5].view(-1, 5)).view(pred_pi.shape[0], pred_pi.shape[1]))
                        #print(box_sides[i].shape)
                    box_sides = torch.stack(box_sides, dim = 2)
                    outputs = torch.cat([box_sides, pred_rest], dim = 2)

                else:
                    outputs = self.model(imgs)
            for imgidx in range(len(outputs)):
                bbox,bboxvari = _postprocess(outputs[imgidx], imgs.shape[-1], ori_shapes[imgidx])
                nms_boxes, nms_scores, nms_labels = torch_nms(self.args.EVAL,bbox,
                                                                 variance=bboxvari)
                if nms_boxes is not None:
                    self.TESTevaluator.append(imgpath[imgidx][0],
                                              nms_boxes.cpu().numpy(),
                                              nms_scores.cpu().numpy(),
                                              nms_labels.cpu().numpy())
        results = self.TESTevaluator.evaluate()
        imgs = self.TESTevaluator.visual_imgs
        for k, v in zip(self.logger_custom, results):
            print("{}:{}".format(k, v))
        return results, imgs
    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG.
        """
        num_samples = 1
        gm = torch.distributions.mixture_same_family.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(probs=pi),
            component_distribution=torch.distributions.Normal(
                loc=torch.squeeze(mu),
                scale=torch.squeeze(sigma)))
        samples = gm.sample(torch.Size([num_samples]))
        return samples
    def mean(self, pi, sigma, mu):
        """Draw samples from a MoG.
        """
        num_samples = 1
        gm = torch.distributions.mixture_same_family.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(probs=pi),
            component_distribution=torch.distributions.Normal(
                loc=torch.squeeze(mu),
                scale=torch.squeeze(sigma)))
        samples = gm.mean
        return samples
