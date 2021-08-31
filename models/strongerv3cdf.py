from models.backbone import *
from models.backbone.helper import *
from models.backbone.baseblock import *

class StrongerV3CDF(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.numclass=cfg.numcls
        self.gt_per_grid=cfg.gt_per_grid
        self.backbone = eval(cfg.backbone)(pretrained=cfg.backbone_pretrained)
        self.outC = self.backbone.backbone_outchannels
        self.heads=[]
        self.activate_type = 'relu6'
        self.headslarge=nn.Sequential(OrderedDict([
            ('conv0',conv_bn(self.outC[0],512,kernel=1,stride=1,padding=0)),
            ('conv1', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv2', conv_bn(1024, 512, kernel=1,stride=1,padding=0)),
            ('conv3', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv4', conv_bn(1024, 512, kernel=1,stride=1,padding=0)   ),
        ]))
        self.detlarge=nn.Sequential(OrderedDict([
            ('conv5',sepconv_bn(512+3*4,1024,kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv6', conv_bias(1024, self.gt_per_grid*(self.numclass+5+4),kernel=1,stride=1,padding=0))
        ]))
        self.mergelarge=nn.Sequential(OrderedDict([
            ('conv7',conv_bn(512,256,kernel=1,stride=1,padding=0)),
            ('upsample0',nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        #-----------------------------------------------
        self.headsmid=nn.Sequential(OrderedDict([
            ('conv8',conv_bn(self.outC[1]+256,256,kernel=1,stride=1,padding=0)),
            ('conv9', sepconv_bn(256, 512, kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv10', conv_bn(512, 256, kernel=1,stride=1,padding=0)),
            ('conv11', sepconv_bn(256, 512, kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv12', conv_bn(512, 256, kernel=1,stride=1,padding=0)),
        ]))
        self.detmid=nn.Sequential(OrderedDict([
            ('conv13',sepconv_bn(256+3*4,512,kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv14', conv_bias(512, self.gt_per_grid*(self.numclass+5+4),kernel=1,stride=1,padding=0))
        ]))
        self.mergemid=nn.Sequential(OrderedDict([
            ('conv15',conv_bn(256,128,kernel=1,stride=1,padding=0)),
            ('upsample0',nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        #-----------------------------------------------
        self.headsmall=nn.Sequential(OrderedDict([
            ('conv16',conv_bn(self.outC[2]+128,128,kernel=1,stride=1,padding=0)),
            ('conv17', sepconv_bn(128, 256, kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv18', conv_bn(256, 128, kernel=1,stride=1,padding=0)),
            ('conv19', sepconv_bn(128, 256, kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv20', conv_bn(256, 128, kernel=1,stride=1,padding=0)),
        ]))
        self.detsmall=nn.Sequential(OrderedDict([
            ('conv21',sepconv_bn(128+3*4,256,kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv22', conv_bias(256, self.gt_per_grid*(self.numclass+5+4),kernel=1,stride=1,padding=0))
        ]))
        if cfg.ASFF:
            self.asff0 = ASFF(0, activate=self.activate_type)
            self.asff1 = ASFF(1, activate=self.activate_type)
            self.asff2 = ASFF(2, activate=self.activate_type)
    def load_partial_state(self,backbone,headslarge, mergelarge, headsmid, mergemid, headsmall):
        self.backbone.load_state_dict(backbone)
        self.headslarge.load_state_dict(headslarge)
        self.mergelarge.load_state_dict(mergelarge)
        #-----------------------------------------------
        self.headsmid.load_state_dict(headsmid)
        self.mergemid.load_state_dict(mergemid)
        #-----------------------------------------------
        self.headsmall.load_state_dict(headsmall)

    def decode(self,output,stride):
        bz=output.shape[0]
        gridsize=output.shape[-1]

        output=output.permute(0,2,3,1)
        output=output.view(bz,gridsize,gridsize,self.gt_per_grid,5+self.numclass+4)
        x1y1,x2y2,variance,conf,prob=torch.split(output,[2,2,4,1,self.numclass],dim=4)

        xyxy=torch.cat((x1y1,x2y2),dim=4)
        xyxy = torch.sigmoid(xyxy)
        conf=torch.sigmoid(conf)
        prob=torch.sigmoid(prob)
        output=torch.cat((xyxy,variance,conf,prob),4)
        return output
    def decode_infer(self,output,stride):
        bz=output.shape[0]
        gridsize=output.shape[-1]

        output=output.permute(0,2,3,1)
        output=output.view(bz,gridsize,gridsize,self.gt_per_grid,5+self.numclass+4)
        x1y1,x2y2,variance,conf,prob=torch.split(output,[2,2,4,1,self.numclass],dim=4)

        xyxy=torch.cat((x1y1,x2y2),dim=4)
        xyxy = torch.sigmoid(xyxy)
        conf=torch.sigmoid(conf)
        prob=torch.sigmoid(prob)
        output=torch.cat((xyxy,variance,conf,prob),4)

        output=output.view(bz,-1,5+self.numclass+4)
        return output
    def forward(self,input, y_large, y_mid, y_small):
        feat_small, feat_mid, feat_large = self.backbone(input)
        conv = self.headslarge(feat_large)
        convlarge=conv

        conv = self.mergelarge(convlarge)
        conv = self.headsmid(torch.cat((conv, feat_mid), dim=1))
        convmid=conv

        conv = self.mergemid(convmid)

        conv = self.headsmall(torch.cat((conv, feat_small), dim=1))
        convsmall=conv
        if self.cfg.ASFF:
            convlarge=self.asff0(convlarge,convmid,convsmall)
            convmid=self.asff1(convlarge,convmid,convsmall)
            convsmall=self.asff2(convlarge,convmid,convsmall)

        combined = torch.cat((convlarge, y_large), 1)
        outlarge = self.detlarge(combined)
        combined = torch.cat((convmid, y_mid), 1)
        outmid = self.detmid(combined)
        combined = torch.cat((convsmall, y_small), 1)
        outsmall = self.detsmall(combined)

        if self.training:
            predlarge = self.decode(outlarge, 32)
            predmid = self.decode(outmid, 16)
            predsmall = self.decode(outsmall, 8)
        else:
            predlarge = self.decode_infer(outlarge, 32)
            predmid = self.decode_infer(outmid, 16)
            predsmall = self.decode_infer(outsmall, 8)
            pred=torch.cat([predsmall,predmid,predlarge],dim=1)
            return pred
        return outsmall,outmid,outlarge,predsmall,predmid,predlarge
    def partial_forward(self, input):
        feat_small, feat_mid, feat_large = self.backbone(input)
        conv = self.headslarge(feat_large)
        convlarge=conv

        conv = self.mergelarge(convlarge)
        conv = self.headsmid(torch.cat((conv, feat_mid), dim=1))
        convmid=conv

        conv = self.mergemid(convmid)

        conv = self.headsmall(torch.cat((conv, feat_small), dim=1))
        convsmall=conv
        if self.cfg.ASFF:
            convlarge=self.asff0(convlarge,convmid,convsmall)
            convmid=self.asff1(convlarge,convmid,convsmall)
            convsmall=self.asff2(convlarge,convmid,convsmall)
        return convlarge, convmid, convsmall
    def partial_forward_2(self, convlarge, convmid, convsmall, outlarge_orig, outmid_orig, outsmall_orig, y_large, y_mid, y_small):
        combined = torch.cat((convlarge, y_large), 1)
        outlarge = self.detlarge(combined)
        combined = torch.cat((convmid, y_mid), 1)
        outmid = self.detmid(combined)
        combined = torch.cat((convsmall, y_small), 1)
        outsmall = self.detsmall(combined)

        predlarge = self.decode_infer(outlarge, 32)
        predmid = self.decode_infer(outmid, 16)
        predsmall = self.decode_infer(outsmall, 8)
        pred=torch.cat([predsmall,predmid,predlarge],dim=1)
        return pred
if __name__ == '__main__':
    import torch.onnx

    # net=YoloV3(20)
    net=YoloV3(0)
    load_tf_weights(net,'cocoweights-half.pkl')

    assert 0
    model=net.eval()
    load_checkpoint(model,torch.load('checkpoints/coco512_prune/checkpoint-best.pth'))
    statedict=model.state_dict()
    layer2block= defaultdict(list)
    for k,v in model.state_dict().items():
        if 'num_batches_tracked' in k:
            statedict.pop(k)
    for idx,(k,v) in enumerate(statedict.items()):
        if 'mobilev2' in k:
            continue
        else:
            flag=k.split('.')[1]
            layer2block[flag].append((k,v))
    for k,v in layer2block.items():
        print(k,len(v))
    pruneratio=0.1

    # #onnx
    # input = torch.randn(1, 3, 320, 320)
    # torch.onnx.export(net, input, "coco320.onnx", verbose=True)
    # #onnxcheck
    # model=onnx.load("coco320.onnx")
    # onnx.checker.check_model(model)
