from models.backbone import *
from models.backbone.helper import *
from models.backbone.baseblock import *

class StrongerV3Quantile(nn.Module):
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
            ('conv5',sepconv_bn(512,1024,kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv6', conv_bias(1024, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0))
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
            ('conv13',sepconv_bn(256,512,kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv14', conv_bias(512, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0))
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
            ('conv21',sepconv_bn(128,256,kernel=3, stride=1, padding=1,seprelu=cfg.seprelu)),
            ('conv22', conv_bias(256, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0))
        ]))
        if cfg.ASFF:
            self.asff0 = ASFF(0, activate=self.activate_type)
            self.asff1 = ASFF(1, activate=self.activate_type)
            self.asff2 = ASFF(2, activate=self.activate_type)
        self.conv_bias_large_orig = conv_bias(1024, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0)
        self.conv_bias_mid_orig = conv_bias(512, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0)
        self.conv_bias_small_orig = conv_bias(256, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0)
        self.sepconv_large = sepconv_bn(512,1024,kernel=3, stride=1, padding=1,seprelu=self.cfg.seprelu)
        self.conv_bias_large =  nn.Sequential(conv_bias(1028, 1024,kernel=1,stride=1,padding=0),
                                nn.LeakyReLU(0.1),
                                nn.BatchNorm2d(1024),
                                conv_bias(1024, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0))
                                #conv_bias(1028, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0)
        self.sepconv_mid = sepconv_bn(256,512,kernel=3, stride=1, padding=1,seprelu=self.cfg.seprelu)
        self.conv_bias_mid =  nn.Sequential(conv_bias(516, 512,kernel=1,stride=1,padding=0),
                            nn.BatchNorm2d(512),
                            nn.LeakyReLU(0.1),
                            conv_bias(512, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0))#conv_bias(516, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0)
        self.sepconv_small = sepconv_bn(128,256,kernel=3, stride=1, padding=1,seprelu=self.cfg.seprelu)
        self.conv_bias_small =  nn.Sequential(conv_bias(260, 256,kernel=1,stride=1,padding=0),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.1),
                                conv_bias(256, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0))#conv_bias(260, self.gt_per_grid*(self.numclass+5),kernel=1,stride=1,padding=0)
    def decode(self,output,stride):
        bz=output.shape[0]
        gridsize=output.shape[-1]

        output=output.permute(0,2,3,1)
        output=output.view(bz,gridsize,gridsize,self.gt_per_grid,5+self.numclass)
        x1y1,x2y2,conf,prob=torch.split(output,[2,2,1,self.numclass],dim=4)
        shiftx=torch.arange(0,gridsize,dtype=torch.float32)
        shifty=torch.arange(0,gridsize,dtype=torch.float32)
        shifty,shiftx=torch.meshgrid([shiftx,shifty])
        shiftx=shiftx.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)
        shifty=shifty.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)

        xy_grid=torch.stack([shiftx,shifty],dim=4).cuda()
        x1y1=(xy_grid+0.5-torch.exp(x1y1))*stride
        x2y2=(xy_grid+0.5+torch.exp(x2y2))*stride

        xyxy=torch.cat((x1y1,x2y2),dim=4)
        conf=torch.sigmoid(conf)
        prob=torch.sigmoid(prob)
        output=torch.cat((xyxy,conf,prob),4)
        return output
    def decode_infer(self,output,stride):
        bz=output.shape[0]
        gridsize=output.shape[-1]

        output=output.permute(0,2,3,1)
        output=output.view(bz,gridsize,gridsize,self.gt_per_grid,5+self.numclass)
        x1y1,x2y2,conf,prob=torch.split(output,[2,2,1,self.numclass],dim=4)

        shiftx=torch.arange(0,gridsize,dtype=torch.float32)
        shifty=torch.arange(0,gridsize,dtype=torch.float32)
        shifty,shiftx=torch.meshgrid([shiftx,shifty])
        shiftx=shiftx.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)
        shifty=shifty.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)

        xy_grid=torch.stack([shiftx,shifty],dim=4).cuda()
        x1y1=(xy_grid+0.5-torch.exp(x1y1))*stride
        x2y2=(xy_grid+0.5+torch.exp(x2y2))*stride

        xyxy=torch.cat((x1y1,x2y2),dim=4)
        conf=torch.sigmoid(conf)
        prob=torch.sigmoid(prob)
        output=torch.cat((xyxy,conf,prob),4)
        output=output.view(bz,-1,5+self.numclass)
        return output

    def forward(self,input, input_alpha):
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
        alpha = input_alpha.repeat(convlarge.size()[2], convlarge.size()[3], 1, 1).permute(2, 3, 0, 1).to(device = 'cuda')
        orig = self.sepconv_large(convlarge)
        combined = torch.cat((orig, alpha), 1)
        outlarge = self.conv_bias_large(combined)
        #outlarge_orig = self.conv_bias_large_orig(orig)
        alpha = input_alpha.repeat(convmid.size()[2], convmid.size()[3], 1, 1).permute(2, 3, 0, 1).to(device = 'cuda')
        orig = self.sepconv_mid(convmid)
        combined = torch.cat((orig, alpha), 1)
        #outmid_orig = self.conv_bias_mid_orig(orig)
        outmid = self.conv_bias_mid(combined)
        alpha = input_alpha.repeat(convsmall.size()[2], convsmall.size()[3], 1, 1).permute(2, 3, 0, 1).to(device = 'cuda')
        orig = self.sepconv_small(convsmall)
        combined = torch.cat((orig, alpha), 1)
        outsmall = self.conv_bias_small(combined)
        #outsmall_orig = self.conv_bias_small_orig(orig)
        if self.training:
            predlarge = self.decode(outlarge, 32)
            predmid=self.decode(outmid,16)
            predsmall=self.decode(outsmall,8)
            # predlarge_orig = self.decode(outlarge_orig, 32)
            # predmid_orig=self.decode(outmid_orig,16)
            # predsmall_orig=self.decode(outsmall_orig,8)
        else:
            predlarge = self.decode_infer(outlarge, 32)
            predmid = self.decode_infer(outmid, 16)
            predsmall = self.decode_infer(outsmall, 8)
            #predlarge_orig = self.decode_infer(outlarge_orig, 32)
            #predmid_orig = self.decode_infer(outmid_orig, 16)
            #predsmall_orig = self.decode_infer(outsmall_orig, 8)
            pred=torch.cat([predsmall,predmid,predlarge],dim=1)
            #pred_orig = torch.cat([predsmall_orig,predmid_orig,predlarge_orig],dim=1)
            return pred
        return outsmall,outmid,outlarge,predsmall,predmid,predlarge#,outsmall_orig, outmid_orig,outlarge_orig, predsmall_orig,predmid_orig,predlarge_orig


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
