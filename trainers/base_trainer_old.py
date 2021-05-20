from utils.util import ensure_dir
from dataset import get_COCO, get_VOC
import os
import time
from dataset import makeImgPyramids
from models.yololoss import yololoss, yololoss_quantile, yololoss_quantile_adjusted
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
import einops
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
                #name=k #remove 'module.' of DataParallel
                name = "module."+k
                new_state_dict[name]=v
            self.model.load_state_dict(new_state_dict)
            #load_checkpoint(self.model,ckptfile)
            if not self.args.finetune and not self.args.do_test and not self.args.Prune.do_test:
                self.optimizer.load_state_dict(ckptfile['opti_dict'])
                self.global_epoch = ckptfile['epoch']
                self.global_iter = ckptfile['iter']
                self.best_mAP = ckptfile['metric']
        print("successfully load checkpoint {}".format(self.args.EXPER.resume))
    def _load_ckpt_quantile_adjusted(self, name):
        cfg2.merge_from_file("configs/strongerv3_asff.yaml")
        net = eval(cfg2.MODEL.modeltype)(cfg=cfg2.MODEL).cuda()
        if self.args.EXPER.resume == "load_voc":
            load_tf_weights(self.model, 'vocweights.pkl')
        else:  # iter or best
            ckptfile = torch.load(os.path.join('./checkpoints/strongerv3_asff/', 'checkpoint-{}.pth'.format(name)))
            state_dict=ckptfile['state_dict']
            new_state_dict= {}
            for k, v in state_dict.items():
                name=k #remove 'module.' of DataParallel
                #name = "module."+k
                new_state_dict[name]=v
            net.load_state_dict(new_state_dict)
            backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall = net.get_info()
            self.model.load_partial_state(backbone,headslarge, detlarge, mergelarge, headsmid, detmid, mergemid, headsmall, detsmall)


    def _get_model(self):
        self.save_path = './checkpoints/{}/'.format(self.experiment_name)
        ensure_dir(self.save_path)
        self._prepare_device()
        if self.args.EXPER.resume:
            self._load_ckpt()
        # if self.args.EXPER.experiment_name == 'strongerv3_quantile_adjusted':
        #     self._load_ckpt_quantile_adjusted("best")

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
            if epoch > 2:
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
            #if epoch % 5 == 0:
            if epoch>10:
                self.model.unfreeze()
            self._save_ckpt(metric=0)

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
            if self.args.EXPER.experiment_name == 'strongerv3_quantile':
                self.train_step_quantile(img, labels)
            if self.args.EXPER.experiment_name == 'strongerv3_quantile_adjusted':
                self.train_step_quantile_adjusted(img, labels)
            else:
                self.train_step(img, labels)
    def _crps_calculate(self, num_samples):
        self.model.eval()
        for sample_num in range(num_samples):
            for i, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
                inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
                img, _, _, *labels = inputs
                if self.args.EXPER.experiment_name == 'strongerv3_quantile':
                    self.crps_step(img, labels)
                if self.args.EXPER.experiment_name == 'strongerv3_quantile_adjusted':
                    self.crps_step(img, labels)
                else:
                    self.crps_step(img, labels)
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
    def train_step_quantile(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        alpha = torch.rand([self.args.OPTIM.batch_size, 4])#torch.full([self.args.OPTIM.batch_size, 4], fill_value = 0)
        #outsmall, outmid, outlarge, predsmall, predmid, predlarge, outsmall_orig, outmid_orig, outlarge_orig, predsmall_orig,predmid_orig,predlarge_orig = self.model(imgs, alpha)
        outsmall, outmid, outlarge, predsmall, predmid, predlarge = self.model(imgs, alpha)
        GIOUloss,conf_loss,probloss = yololoss_quantile(self.args.MODEL,outsmall, outmid, outlarge, #outsmall_orig, outmid_orig, outlarge_orig,
                            predsmall, predmid, predlarge,
                             #predsmall_orig, predmid_orig, predlarge_orig,
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
    def train_step_quantile_adjusted(self, imgs, labels):
        imgs = imgs.cuda()
        labels = [label.cuda() for label in labels]
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = labels
        #print(label_sbbox.view(bz,-1,5+self.numclass))
        alpha = torch.rand([self.args.OPTIM.batch_size, 4])
        outsmall, outmid, outlarge, predsmall, predmid, predlarge, outsmall_orig, outmid_orig, outlarge_orig, predsmall_orig,predmid_orig,predlarge_orig = self.model(imgs, alpha)
        GIOUloss,conf_loss,probloss = yololoss_quantile_adjusted(self.args.MODEL,outsmall, outmid, outlarge, outsmall_orig, outmid_orig, outlarge_orig,
                            predsmall, predmid, predlarge,
                             predsmall_orig, predmid_orig, predlarge_orig,
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
    def sample_quantile(self, num_samples,validiter=-1):
      def _postprocess(pred_bbox, test_input_size, org_img_shape):
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
          return bboxes
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
              convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
          alpha = torch.full([self.args.OPTIM.batch_size, 4], fill_value = 0.5)
          with torch.no_grad():
              outputs = self.model.module.partial_forward_2(convlarge, convmid, convsmall, alpha)
          bbox_array = [outputs.cpu()]
          for i in range(num_samples):
              alpha = torch.rand([self.args.OPTIM.batch_size, 4])
              with torch.no_grad():
                  outputs = self.model.module.partial_forward_2(convlarge, convmid, convsmall, alpha).cpu()
                  bbox_array.append(outputs)
          bbox_final = torch.stack(bbox_array, dim = 3)
          for imgidx in range(len(outputs)):
              postprocessed_array = []
              for i in range(num_samples+1):
                  postprocessed_array.append(_postprocess(bbox_final[imgidx, :, :, i], imgs.shape[-1], ori_shapes[imgidx].cpu()))
              bbox_image = torch.stack(postprocessed_array, dim = 2)
              nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox_image,
                                                               variance=None)
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
    def sample_gaussian(self,num_samples, validiter=-1):
        def _postprocess(pred_bbox, test_input_size, org_img_shape):
            if self.args.MODEL.boxloss == 'KL':
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
        s = time.time()
        self.model.eval()
        for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
            if idx_batch == validiter:  # to save time
                break
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            (imgs, imgpath, ori_shapes, *_) = inputs
            imgs = imgs.cuda()
            ori_shapes = ori_shapes.cuda()
            alpha = torch.full([self.args.OPTIM.batch_size, 4], fill_value = 0.5)
            with torch.no_grad():
                if self.args.EXPER.experiment_name == "strongerv3_quantile":
                    outputs = self.model(imgs, alpha)
                elif self.args.EXPER.experiment_name == "strongerv3_quantile_adjusted":
                    convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
                    outputs = self.model.module.partial_forward_2(convlarge, convmid, convsmall, alpha)
                else:
                    outputs = self.model(imgs)
            for imgidx in range(len(outputs)):
                bbox,bboxvari = _postprocess(outputs[imgidx], imgs.shape[-1], ori_shapes[imgidx])
                bbox = einops.repeat(bbox, 'm n -> m n k', k=(num_samples+1))
                bbox[:, 0:4, 1:] = torch.normal(bbox[:, 0:4, 1:], einops.repeat(bboxvari, 'm n -> m n k', k=num_samples))
                nms_boxes, nms_scores, nms_labels = torch_nms_sampling(self.args.EVAL,bbox,
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
    def _valid_epoch(self,validiter=-1):
        def _postprocess(pred_bbox, test_input_size, org_img_shape):
            if self.args.MODEL.boxloss == 'KL':
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
        evaluation_method = "crps"
        if evaluation_method == "crps":
            return self.sample_quantile(200, validiter)
        s = time.time()
        self.model.eval()
        for idx_batch, inputs in tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader)):
            if idx_batch == validiter:  # to save time
                break
            inputs = [input if isinstance(input, list) else input.squeeze(0) for input in inputs]
            (imgs, imgpath, ori_shapes, *_) = inputs
            imgs = imgs.cuda()
            ori_shapes = ori_shapes.cuda()
            alpha = torch.full([self.args.OPTIM.batch_size, 4], fill_value = 0.5)
            with torch.no_grad():
                if self.args.EXPER.experiment_name == "strongerv3_quantile":
                    outputs = self.model(imgs, alpha)
                elif self.args.EXPER.experiment_name == "strongerv3_quantile_adjusted":
                    convlarge, convmid, convsmall = self.model.module.partial_forward(imgs)
                    outputs = self.model.module.partial_forward_2(convlarge, convmid, convsmall, alpha)
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
