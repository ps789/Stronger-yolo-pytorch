import torch
import torch.nn.functional as F
import utils.GIOU as GIOUloss
from torch.distributions import Categorical
from torch.autograd import Variable
import numpy as np
strides = [8, 16, 32]


def focalloss(target, actual, alpha=1, gamma=2):
    focal = alpha * torch.pow(torch.abs(target - actual), gamma)
    return focal

def yololoss(
        cfg,conv_sbbox, conv_mbbox, conv_lbbox,
        pred_sbbox, pred_mbbox, pred_lbbox,
        label_sbbox, label_mbbox, label_lbbox,
        sbboxes, mbboxes, lbboxes):
    """
    :param conv_sbbox: shap e为(batch_size, image_size / 8, image_size / 8, anchors_per_scale * (5 + num_classes))
    :param conv_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale * (5 + num_classes))
    :param conv_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale * (5 + num_classes))
    :param pred_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale, (5 + num_classes))
    :param pred_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale, (5 + num_classes))
    :param pred_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale, (5 + num_classes))
    :param label_sbbox: shape为(batch_size, input_size / 8, input_size / 8, anchor_per_scale, 6 + num_classes)
    :param label_mbbox: shape为(batch_size, input_size / 16, input_size / 16, anchor_per_scale, 6 + num_classes)
    :param label_lbbox: shape为(batch_size, input_size / 32, input_size / 32, anchor_per_scale, 6 + num_classes)
    :param sbboxes: shape为(batch_size, max_bbox_per_scale, 4)
    :param mbboxes: shape为(batch_size, max_bbox_per_scale, 4)
    :param lbboxes: shape为(batch_size, max_bbox_per_scale, 4)
    :return:
    """
    GIOUloss_s,conf_loss_s,probloss_s = loss_per_scale(conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                strides[0],cfg=cfg)
    GIOUloss_m,conf_loss_m,probloss_m = loss_per_scale(conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
                                strides[1],cfg=cfg)
    GIOUloss_l,conf_loss_l,probloss_l = loss_per_scale(conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
                                strides[2],cfg=cfg)
    GIOUloss=GIOUloss_s+GIOUloss_m+GIOUloss_l
    conf_loss=conf_loss_s+conf_loss_m+conf_loss_l
    probloss=probloss_s+probloss_m+probloss_l
    return GIOUloss,conf_loss,probloss


def loss_per_scale(conv, pred, label, bboxes, stride,cfg):
    """
    :param name: loss的名字
    :param conv: conv是yolo卷积层的原始输出
    shape为(batch_size, output_size, output_size, anchor_per_scale * (5 + num_class))
    :param pred: conv是yolo输出的预测bbox的信息(x, y, w, h, conf, prob)，
    其中(x, y, w, h)的大小是相对于input_size的，如input_size=416，(x, y, w, h) = (120, 200, 50, 70)
    shape为(batch_size, output_size, output_size, anchor_per_scale, 5 + num_class)
    :param label: shape为(batch_size, output_size, output_size, anchor_per_scale, 6 + num_classes)
    只有负责预测GT的对应位置的数据才为(xmin, ymin, xmax, ymax, 1, classes, mixup_weights),
    其他位置的数据都为(0, 0, 0, 0, 0, 0..., 1)
    :param bboxes: shape为(batch_size, max_bbox_per_scale, 4)，
    存储的坐标为(xmin, ymin, xmax, ymax)
    bboxes用于计算相应detector的预测框与该detector负责预测的所有bbox的IOU
    :param anchors: 相应detector的anchors
    :param stride: 相应detector的stride
    """
    bcelogit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

    conv = conv.permute(0, 2, 3, 1)
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    numanchor = cfg.gt_per_grid
    conv = conv.view(batch_size, output_size, output_size,numanchor, -1)
    if cfg.boxloss == 'KL' or cfg.boxloss == 'quantile' or cfg.boxloss == 'cdf':
        conv_raw_conf = conv[..., 8:9]
        conv_raw_prob = conv[..., 9:]

        pred_coor = pred[..., 0:4]
        pred_vari = pred[..., 4:8]
        pred_conf = pred[..., 8:9]
        pred_prob = pred[..., 9:]
    elif cfg.boxloss == 'MDN':
        conv_raw_conf = conv[..., 60:61]
        conv_raw_prob = conv[..., 61:]
        pred_pi = pred[..., 0:20]
        pred_coor = pred[..., 20:40]
        pred_vari = pred[..., 40:60]
        pred_conf = pred[..., 60:61]
        pred_prob = pred[..., 61:]

    else:
        conv_raw_conf = conv[..., 4:5]
        conv_raw_prob = conv[..., 5:]

        pred_prob = pred[..., 5:]
        pred_coor = pred[..., 0:4]
        pred_conf = pred[..., 4:5]

    label_coor = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    label_prob = label[..., 5:-1]
    label_mixw = label[..., -1:]
    # 计算GIOU损失
    bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
    bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (input_size ** 2)
    if cfg.boxloss == 'iou':
        giou = GIOUloss.GIOU(pred_coor, label_coor).unsqueeze(-1)
        giou_loss = respond_bbox * bbox_loss_scale * (1.0 - giou)
        bbox_loss = giou_loss
    elif cfg.boxloss == 'l1':
        l1_loss = respond_bbox * bbox_loss_scale * smooth_loss(target=label_coor, input=pred_coor) * cfg.l1scale
        bbox_loss = l1_loss
    elif cfg.boxloss == 'KL':
        l1_loss = respond_bbox * bbox_loss_scale * (
                torch.exp(-pred_vari) * smooth_loss(target=label_coor, input=pred_coor) + 0.5 * pred_vari) * cfg.l1scale
        bbox_loss = l1_loss
    elif cfg.boxloss=='diou':
        diou = GIOUloss.DIOU(pred_coor, label_coor).unsqueeze(-1)
        diou_loss = respond_bbox * bbox_loss_scale * (1.0 - diou)
        bbox_loss = diou_loss
    elif cfg.boxloss=="MDN":
        log_likelihood = 0
        box_sides = []
        box_loss = []
        mdn_size = 5
        for i in range(4):
            # gm = torch.distributions.mixture_same_family.MixtureSameFamily(
            # mixture_distribution=torch.distributions.Categorical(probs=pred_pi[..., i*mdn_size:i*mdn_size+mdn_size]),
            # component_distribution=torch.distributions.Normal(
            #     loc=torch.squeeze(pred_coor[..., i*mdn_size:i*mdn_size+mdn_size]),
            #     scale=torch.squeeze(torch.exp(-pred_vari[..., i*mdn_size:i*mdn_size+mdn_size]))))
            # log_likelihood = -gm.log_prob(label_coor[..., i])
            exponent = -0.5 * torch.square((label_coor[..., i].unsqueeze(-1) - pred_coor[..., i*mdn_size:i*mdn_size+mdn_size])) / (1e-4+torch.exp(-pred_vari[..., i*mdn_size:i*mdn_size+mdn_size]))
            exponent = exponent + torch.log(1e-4+pred_pi[..., i*mdn_size:i*mdn_size+mdn_size]) - (0.5)*(np.log(2*np.pi)-pred_vari[..., i*mdn_size:i*mdn_size+mdn_size])
            max_exponent = torch.max(exponent, dim = -1, keepdims=True)[0]
            mod_exponent = exponent - max_exponent
            gauss_mix = torch.sum(torch.exp(mod_exponent), axis = -1)
            log_likelihood = -max_exponent.squeeze() - torch.log(1e-4+gauss_mix)
            box_loss.append(log_likelihood)
            box_sides.append(means(pred_pi[..., i*mdn_size:i*mdn_size+mdn_size].view(-1, mdn_size), torch.exp(-pred_vari[..., i*mdn_size:i*mdn_size+mdn_size]).view(-1, mdn_size), pred_coor[..., i*mdn_size:i*mdn_size+mdn_size].view(-1, mdn_size)).view(pred_pi.shape[0], pred_pi.shape[1], pred_pi.shape[2], pred_pi.shape[3]))

            #print(box_sides[i].shape)
        box_loss = torch.stack(box_loss, dim = 4)
        box_sides = torch.stack(box_sides, dim = 4)
        #sample(pred_pi.view(-1, 5, 4), pred_vari.view(-1, 5, 4), pred_coor.view(-1, 5, 4))
        #getSamples(pred_pi.view(-1, 5, 4).cpu().detach().numpy(), pred_vari.view(-1, 5, 4).cpu().detach().numpy(), pred_coor.view(-1, 5, 4).cpu().detach().numpy())
        bbox_loss = respond_bbox * bbox_loss_scale * box_loss * cfg.l1scale
    else:
        l1_loss = respond_bbox * bbox_loss_scale * (
                torch.exp(-pred_vari) * smooth_loss(target=label_coor, input=pred_coor) + 0.5 * pred_vari) * cfg.l1scale
        bbox_loss = l1_loss
        #raise NotImplementedError
    bbox_loss=bbox_loss*label_mixw
    # (2)计算confidence损失
    if cfg.boxloss=="MDN":
        iou = GIOUloss.iou_calc3(box_sides.unsqueeze(4),bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
    else:
        iou = GIOUloss.iou_calc3(pred_coor.unsqueeze(4),bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
    max_iou,_=torch.max(iou,dim=-1)

    max_iou = max_iou.unsqueeze(-1)
    respond_bgd = (1.0 - respond_bbox) * (max_iou < 0.5).float()

    conf_focal = focalloss(respond_bbox, pred_conf)

    conf_loss = conf_focal * (
            respond_bbox * bcelogit_loss(target=respond_bbox, input=conv_raw_conf)
            +
            respond_bgd * bcelogit_loss(target=respond_bbox, input=conv_raw_conf)
    )
    conf_loss=conf_loss*label_mixw
    # (3)计算classes损失
    if conv_raw_prob.shape[-1]!=0:
        if cfg.clsfocal:
            cls_focal = focalloss(label_prob, pred_prob)
            prob_loss = cls_focal * respond_bbox * bcelogit_loss(target=label_prob, input=conv_raw_prob)
        else:
            prob_loss = respond_bbox * bcelogit_loss(target=label_prob, input=conv_raw_prob)
    else:
        prob_loss = torch.zeros_like(label_prob)
    prob_loss=prob_loss*label_mixw
    return bbox_loss.sum(),conf_loss.sum(),prob_loss.sum()
    # loss = tf.concat([GIOU_loss, conf_loss, prob_loss], axis=-1)
    # loss = torch.cat([GIOU_loss, conf_loss], dim=-1)
    # loss = loss * label_mixw
    # loss = torch.sum(loss)
    # return loss

def yololoss_cdf(cfg,conv_sbbox, conv_mbbox, conv_lbbox,
pred_sbbox, pred_mbbox, pred_lbbox,
label_sbbox, label_mbbox, label_lbbox,
sbboxes, mbboxes, lbboxes, y_large, y_mid, y_small):
    GIOUloss_s,conf_loss_s,probloss_s = cdfloss(conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                strides[0],cfg=cfg, samples = y_small)
    GIOUloss_m,conf_loss_m,probloss_m = cdfloss(conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
        strides[1],cfg=cfg, samples = y_mid)
    GIOUloss_l,conf_loss_l,probloss_l = cdfloss(conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
        strides[2],cfg=cfg, samples = y_large)
    GIOUloss=(GIOUloss_s+GIOUloss_m+GIOUloss_l)
    conf_loss=conf_loss_s+conf_loss_m+conf_loss_l
    probloss=probloss_s+probloss_m+probloss_l
    return GIOUloss,conf_loss,probloss
def yololoss_cdf_gaussian(cfg,conv_sbbox, conv_mbbox, conv_lbbox,
pred_sbbox, pred_mbbox, pred_lbbox,
label_sbbox, label_mbbox, label_lbbox,
sbboxes, mbboxes, lbboxes, y_large, y_mid, y_small):
    GIOUloss_s = cdfloss_gaussian(conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                strides[0],cfg=cfg, samples = y_small)
    GIOUloss_m = cdfloss_gaussian(conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
        strides[1],cfg=cfg, samples = y_mid)
    GIOUloss_l = cdfloss_gaussian(conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
        strides[2],cfg=cfg, samples = y_large)
    GIOUloss=(GIOUloss_s+GIOUloss_m+GIOUloss_l)
    return GIOUloss
def cdfloss_gaussian(conv, pred, label, bboxes, stride, cfg, samples):
    conv = conv.permute(0, 2, 3, 1)
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    numanchor = cfg.gt_per_grid
    conv = conv.view(batch_size, output_size, output_size,numanchor, -1)
    if cfg.boxloss == 'KL' or cfg.boxloss == 'quantile' or cfg.boxloss == 'cdf':
        conv_raw_conf = conv[..., 8:9]
        conv_raw_prob = conv[..., 9:]

        pred_coor = pred[..., 0:4]
        pred_vari = pred[..., 4:8]
        pred_conf = pred[..., 8:9]
        pred_prob = pred[..., 9:]
    label_coor = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    label_prob = label[..., 5:-1]
    label_mixw = label[..., -1:]
    probabilities = torch.distributions.normal.Normal(pred_coor, pred_vari).cdf(samples)
    indicator = torch.where(samples - label_coor >= 0, 1, 0)
    losses = (probabilities - indicator) ** 2
    l1_loss = respond_bbox * (losses) #* cfg.l1scale * bbox_loss_scale
    #print(torch.sum(l1_loss))
    #print(pred_coor[torch.repeat_interleave(respond_bbox, 4, axis = 4) != 0])
    bbox_loss = l1_loss
    bbox_loss=bbox_loss*label_mixw
    return(bbox_loss.sum())


def cdfloss(conv, pred, label, bboxes, stride, cfg, samples):
    bcelogit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

    conv = conv.permute(0, 2, 3, 1)
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    numanchor = cfg.gt_per_grid
    conv = conv.view(batch_size, output_size, output_size,numanchor, -1)

    conv_raw_conf = conv[..., 8:9]
    conv_raw_prob = conv[..., 9:]

    pred_coor = pred[..., 0:4]
    pred_vari = pred[..., 4:8]
    pred_conf = pred[..., 8:9]
    pred_prob = pred[..., 9:]

    label_coor = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    label_prob = label[..., 5:-1]
    label_mixw = label[..., -1:]
    bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
    bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (input_size ** 2)
    #print(samples[torch.repeat_interleave(respond_bbox, 4, axis = 4) != 0])
    #print(label_coor[torch.repeat_interleave(respond_bbox, 4, axis = 4) != 0])
    indicator = torch.where(samples - label_coor >= 0, 1, 0)
    losses = (pred_coor - indicator) ** 2
    l1_loss = respond_bbox * (losses) #* cfg.l1scale * bbox_loss_scale
    #print(torch.sum(l1_loss))
    #print(pred_coor[torch.repeat_interleave(respond_bbox, 4, axis = 4) != 0])
    bbox_loss = l1_loss
    bbox_loss=bbox_loss*label_mixw
    # (2)计算confidence损失
    iou = GIOUloss.iou_calc3(samples.unsqueeze(4),bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
    max_iou,_=torch.max(iou,dim=-1)

    max_iou = max_iou.unsqueeze(-1)
    respond_bgd = (1.0 - respond_bbox) * (max_iou < 0.5).float()

    conf_focal = focalloss(respond_bbox, pred_conf)

    conf_loss = conf_focal * (
            respond_bbox * bcelogit_loss(target=respond_bbox, input=conv_raw_conf)
            +
            respond_bgd * bcelogit_loss(target=respond_bbox, input=conv_raw_conf)
    )
    conf_loss=conf_loss*label_mixw
    # (3)计算classes损失
    if conv_raw_prob.shape[-1]!=0:
        if cfg.clsfocal:
            cls_focal = focalloss(label_prob, pred_prob)
            prob_loss = cls_focal * respond_bbox * bcelogit_loss(target=label_prob, input=conv_raw_prob)
        else:
            prob_loss = respond_bbox * bcelogit_loss(target=label_prob, input=conv_raw_prob)
    else:
        prob_loss = torch.zeros_like(label_prob)
    prob_loss=prob_loss*label_mixw
    return bbox_loss.sum(),conf_loss.sum(),prob_loss.sum()
def yololoss_quantile_adjusted(
        cfg,conv_sbbox, conv_mbbox, conv_lbbox,
        pred_sbbox, pred_mbbox, pred_lbbox,
        label_sbbox, label_mbbox, label_lbbox,
        sbboxes, mbboxes, lbboxes, alpha):
    """
    :param conv_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale * (5 + num_classes))
    :param conv_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale * (5 + num_classes))
    :param conv_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale * (5 + num_classes))
    :param pred_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale, (5 + num_classes))
    :param pred_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale, (5 + num_classes))
    :param pred_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale, (5 + num_classes))
    :param label_sbbox: shape为(batch_size, input_size / 8, input_size / 8, anchor_per_scale, 6 + num_classes)
    :param label_mbbox: shape为(batch_size, input_size / 16, input_size / 16, anchor_per_scale, 6 + num_classes)
    :param label_lbbox: shape为(batch_size, input_size / 32, input_size / 32, anchor_per_scale, 6 + num_classes)
    :param sbboxes: shape为(batch_size, max_bbox_per_scale, 4)
    :param mbboxes: shape为(batch_size, max_bbox_per_scale, 4)
    :param lbboxes: shape为(batch_size, max_bbox_per_scale, 4)
    :return:
    """

    GIOUloss_s,conf_loss_s,probloss_s = loss_per_scale(conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                strides[0],cfg=cfg)
    GIOUloss_m,conf_loss_m,probloss_m = loss_per_scale(conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
                                strides[1],cfg=cfg)
    GIOUloss_l,conf_loss_l,probloss_l = loss_per_scale(conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
                                strides[2],cfg=cfg)
    GIOUloss_s = quantileloss(conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                strides[0],cfg=cfg, alphas = alpha)
    GIOUloss_m = quantileloss(conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
        strides[1],cfg=cfg, alphas = alpha)
    GIOUloss_l = quantileloss(conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
        strides[2],cfg=cfg, alphas = alpha)
    GIOUloss=(GIOUloss_s+GIOUloss_m+GIOUloss_l)/200
    conf_loss=conf_loss_s+conf_loss_m+conf_loss_l
    probloss=probloss_s+probloss_m+probloss_l
    return GIOUloss,conf_loss,probloss
def quantileloss(conv, pred, label, bboxes, stride, cfg, alphas):
    bcelogit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

    conv = conv.permute(0, 2, 3, 1)
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    numanchor = cfg.gt_per_grid
    conv = conv.view(batch_size, output_size, output_size,numanchor, -1)

    pred_coor = pred[..., 0:4]

    label_coor = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
    bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (input_size ** 2)
    alpha = alphas.repeat(pred_coor.size()[1], pred_coor.size()[2], pred_coor.size()[3], 1, 1).permute(3, 0, 1, 2, 4).to(device = "cuda")
    errors = label_coor-pred_coor
    losses = torch.max((alpha-1) * errors, alpha * errors)
    l1_loss = respond_bbox * (losses) * cfg.l1scale * bbox_loss_scale
    bbox_loss = l1_loss.sum()
    return bbox_loss
def transform_crps(conv, pred, label, cfg):
    conv = conv.permute(0, 2, 3, 1)
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    numanchor = cfg.gt_per_grid
    conv = conv.view(batch_size, output_size, output_size,numanchor, -1)
    pred_coor = pred[..., 0:4]

    label_coor = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    pred_coor_new = (respond_bbox*pred_coor).flatten()
    label_coor_new = (respond_bbox*label_coor).flatten()
    return pred_coor_new, label_coor_new
def sample(pi, sigma, mu):
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
def means(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    gm = torch.distributions.mixture_same_family.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(probs=pi),
        component_distribution=torch.distributions.Normal(
            loc=torch.squeeze(mu),
            scale=torch.squeeze(sigma)))
    return gm.mean

def getSamples(pis, sigmas, mus):
    l = len(pis[0])
    samples = np.empty((pis.shape[0], 30))
    for i in range(len(pis)):
        choices = np.random.choice(l, 30, p=pis[i])
        distributions = [np.random.normal(mean, std, 30) for mean, std in zip(mus[i], sigmas[i])]
        temporary = []
        for c in range(len(choices)):
          s = np.random.choice(distributions[choices[c]], 1)
          temporary.append(s[0])
        temporary = sorted(temporary)
        samples[i] = temporary
    return samples
