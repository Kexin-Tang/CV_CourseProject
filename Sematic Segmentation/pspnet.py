import torch
from torch import nn
from torch.nn import functional as F
from backbone_8stride import resnet18

'''
    @ PSP module
    
    input:  features        ->  input of encoder
            out_features    ->  output of decoder
            sizes           ->  different pyrimaid size 
'''
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


'''
    @ x2 bilinear upsampling
'''
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


'''
    @ define the PSPNet via Res18
'''
class PSPNet(nn.Module):
    def __init__(self, n_classes=6, sizes=(1, 2, 3, 6), psp_size=512, pretrained=True):
        super().__init__()
        self.feats = resnet18(pretrained=pretrained)
        self.psp = PSPModule(psp_size, 256, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(256, 128)
        self.up_2 = PSPUpsample(128, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda: self.cuda()

    def forward(self, x):
        n, c, h, w = x.size()
        if self.use_cuda:
            x = x.cuda()
        f = self.feats(x)[0]

        up_f = f
        p = self.psp(up_f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        p = F.interpolate(p, (h, w), mode='bilinear', align_corners=True)

        return self.final(p)
