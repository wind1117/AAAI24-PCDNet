import requests
import numpy as np
import pandas as pd
from copy import copy
from pathlib import Path
from copy import deepcopy
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import init
from torchvision.transforms import functional as T
from torch.nn import functional as F
from torch.cuda import amp

from utils.general import non_max_suppression, xyxy2xywh, make_divisible, scale_coords
from utils.torch_utils import time_synchronized
from utils.dataset import letterbox

import sys

sys.path.append(str(Path(__file__).parent))


class ScharrEdge(nn.Module):
    def __init__(self):
        super(ScharrEdge, self).__init__()
        self.x_kernel = torch.Tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ]).reshape(1, 1, 3, 3)
        self.y_kernel = torch.Tensor([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ]).reshape(1, 1, 3, 3)

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        self.x_kernel = self.x_kernel.to(device)
        self.y_kernel = self.y_kernel.to(device)
        self.x_kernel = self.x_kernel.half() if isinstance(x, torch.cuda.HalfTensor) else self.x_kernel
        self.y_kernel = self.y_kernel.half() if isinstance(x, torch.cuda.HalfTensor) else self.y_kernel
        gray_x = T.rgb_to_grayscale(x, num_output_channels=1)
        x_x = F.conv2d(gray_x, self.x_kernel, stride=1, padding=1)
        y_x = F.conv2d(gray_x, self.y_kernel, stride=1, padding=1)
        e_x = torch.sqrt(torch.pow(x_x, 2) + torch.pow(y_x, 2))
        r_x = norm01_in_sample(e_x)
        return r_x


class FusePol(nn.Module):
    def __init__(self, in_channel=3, pool_size=5):
        super(FusePol, self).__init__()
        self.conv_sattn = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.degree_max_map = nn.Sequential(
            Conv(c1=in_channel, c2=in_channel, k=3, s=1),
            nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2),
            nn.Sigmoid()
        )
        self.edge = ScharrEdge()
        self.conv_aop1 = Conv(c1=in_channel, c2=in_channel, k=3, s=1)
        self.conv_dop1 = Conv(c1=in_channel, c2=in_channel, k=3, s=1)
        self.conv_pol1 = Conv(c1=in_channel * 2, c2=in_channel, k=3, s=1)

    def forward(self, x):
        aop, dop = x
        _max, _ = torch.max(dop, dim=1, keepdim=True)
        _avg = torch.mean(dop, dim=1, keepdim=True)
        w_aop = aop * (self.conv_sattn(torch.cat([_max, _avg], dim=1)) + self.degree_max_map(dop))
        w_dop = dop + self.edge(dop)
        pol = self.conv_pol1(torch.cat([self.conv_aop1(w_aop), self.conv_dop1(w_dop)], dim=1))
        return pol


class CDDS3F(nn.Module):
    # cross domain demand supplement
    def __init__(self, in_channel, ca_reduction=16, sa_kernel=7):
        super(CDDS3F, self).__init__()
        self.rgb_admp = nn.AdaptiveMaxPool2d(1)
        self.rgb_adap = nn.AdaptiveAvgPool2d(1)
        self.rgb_se = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ca_reduction, 1, bias=False), nn.SiLU(),
            nn.Conv2d(in_channel // ca_reduction, in_channel, 1, bias=False)
        )
        self.rgb_sa = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=sa_kernel, padding=sa_kernel // 2)

        self.sa_map2pol = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            Conv(c1=1, c2=1, k=3, s=1),
        )

        self.aap_rgb = nn.AdaptiveAvgPool2d(1)
        self.aap_pol = nn.AdaptiveAvgPool2d(1)

        self.c_fc = nn.Sequential(
            nn.Linear(in_channel * 2, in_channel * 2, bias=False),
            nn.SiLU(),
            nn.Linear(in_channel * 2, in_channel * 2, bias=False),
            nn.Sigmoid(),
        )

        self.mix_conv = Conv(in_channel * 2, in_channel, k=1, s=1)

    def forward(self, x):
        rgb, pol = x

        ca_rgb = rgb * torch.sigmoid(self.rgb_se(self.rgb_admp(rgb)) + self.rgb_se(self.rgb_adap(rgb)))
        max_rgb, _ = torch.max(ca_rgb, dim=1, keepdim=True)
        avg_rgb = torch.mean(ca_rgb, dim=1, keepdim=True)
        sa_rgb_map = torch.sigmoid(self.rgb_sa(torch.cat([max_rgb, avg_rgb], dim=1)))
        enh_rgb = rgb + ca_rgb * sa_rgb_map

        enh_pol = pol + pol * self.sa_map2pol(sa_rgb_map)

        b, c, h, w = rgb.shape
        ch_w = self.c_fc(torch.cat([self.aap_rgb(enh_rgb).view(b, c), self.aap_pol(enh_pol).view(b, c)], dim=1))
        ch_wr, ch_wp = ch_w.unsqueeze(-1).split(c, dim=1)
        ch_wr, ch_wp = F.softmax(torch.cat([ch_wr, ch_wp], dim=-1), dim=-1).split(1, dim=-1)
        ch_wr = ch_wr.unsqueeze(-1)
        ch_wp = ch_wp.unsqueeze(-1)

        return self.mix_conv(torch.cat([enh_rgb * ch_wr, enh_pol * ch_wp], dim=1))


class MPM4(nn.Module):
    # material perception memory
    def __init__(self, in_dim):
        super(MPM4, self).__init__()
        self.c1 = in_dim // 2
        self.mk1 = nn.Sequential(
            nn.Conv2d(in_dim, self.c1, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=self.c1), nn.SiLU()
        )
        self.ch_pool = nn.Sequential(
            Conv(c1=self.c1, c2=self.c1, k=1, s=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ch_fc = nn.Sequential(
            nn.Linear(in_features=self.c1, out_features=self.c1 // 2, bias=False),
            nn.SiLU(),
            nn.Linear(in_features=self.c1 // 2, out_features=self.c1, bias=False),
            nn.Sigmoid()
        )
        self.mv2 = nn.Sequential(
            nn.ConvTranspose2d(self.c1, in_dim, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=in_dim), nn.SiLU()
        )

    def forward(self, x):
        _, _, h1, w1 = x.shape
        out = self.mk1(x)

        b2, c2, h2, w2 = out.shape
        out = out + out * self.ch_fc(self.ch_pool(out).view(b2, c2)).unsqueeze(-1).unsqueeze(-1)

        out = self.mv2(out)
        out = F.interpolate(out, size=(h1, w1), mode='bilinear', align_corners=False)
        return out


class MPM3(nn.Module):
    # material perception memory
    def __init__(self, in_dim):
        super(MPM3, self).__init__()
        self.c1 = in_dim // 2
        self.c2 = in_dim // 4
        self.mk1 = nn.Sequential(
            nn.Conv2d(in_dim, self.c1, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=self.c1), nn.SiLU()
        )
        self.mk2 = nn.Sequential(
            nn.Conv2d(self.c1, self.c2, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=self.c2), nn.SiLU()
        )
        self.mv1 = nn.Sequential(
            nn.ConvTranspose2d(self.c2, self.c1, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.c1), nn.SiLU()
        )
        self.mv2 = nn.Sequential(
            nn.ConvTranspose2d(self.c1, in_dim, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=in_dim), nn.SiLU()
        )

    def forward(self, x):
        _, _, h1, w1 = x.shape
        out = self.mk1(x)
        _, _, h2, w2 = out.shape
        out = self.mk2(out)

        out = self.mv1(out)
        out = F.interpolate(out, size=(h2, w2), mode='bilinear', align_corners=False)
        out = self.mv2(out)
        out = F.interpolate(out, size=(h1, w1), mode='bilinear', align_corners=False)
        return out


class FeatureMul(nn.Module):
    def __init__(self):
        super(FeatureMul, self).__init__()

    @torch.no_grad()
    def forward(self, x):
        return x[0] * x[1]


class FeatureSum(nn.Module):
    def __init__(self):
        super(FeatureSum, self).__init__()

    @torch.no_grad()
    def forward(self, x):
        return torch.stack(x, dim=0).sum(dim=0)


class IndexDomains(nn.Module):
    def __init__(self, idx):
        super(IndexDomains, self).__init__()
        self.idx = idx

    @torch.no_grad()
    def forward(self, x):
        return x[self.idx]


def norm01_in_sample(input):
    if len(input.shape) == 3:
        rst = (input - torch.min(input)) / (torch.max(input) - torch.min(input) + 1e-6)
        return rst
    else:
        for i, sample in enumerate(input):
            sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample) + 1e-6)
            input[i] = sample
        return input


def norm01_in_batch(input):
    return (input - torch.min(input)) / (torch.max(input) - torch.min(input) + 1e-6)


def autopad(k, p=None):  # Pad to 'same', (kernel, pading)
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class AvgP(nn.Module):
    def __init__(self, k=3, s=1):
        super(AvgP, self).__init__()
        self.m = nn.AvgPool2d(kernel_size=k, stride=s, padding=k // 2, count_include_pad=False)

    def forward(self, x):
        return self.m(x)


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):  # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3, 'k is {}'.format(k)
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class SPPCSPC(nn.Module):
    #  CSP https://github.com/WongKinYiu/CrossStagePartialNetworks

    def __init__(self, in_channel, out_channel, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * out_channel * e)  # hidden channels
        self.cv1 = Conv(in_channel, c_, 1, 1)
        self.cv2 = Conv(in_channel, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, out_channel, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust models wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # models already converted to models.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    # def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
    #     colors = color_list()
    #     for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
    #         str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
    #         if pred is not None:
    #             for c in pred[:, -1].unique():
    #                 n = (pred[:, -1] == c).sum()  # detections per class
    #                 str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #             if show or save or render:
    #                 for *box, conf, cls in pred:  # xyxy, confidence, class
    #                     label = f'{self.names[int(cls)]} {conf:.2f}'
    #                     plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
    #         img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
    #         if pprint:
    #             print(str.rstrip(', '))
    #         if show:
    #             img.show(self.files[i])  # show
    #         if save:
    #             f = self.files[i]
    #             img.save(Path(save_dir) / f)  # save
    #             print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
    #         if render:
    #             self.imgs[i] = np.asarray(img)
    #
    # def print(self):
    #     self.display(pprint=True)  # print results
    #     print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)
    #
    # def show(self):
    #     self.display(show=True)  # show results
    #
    # def save(self, save_dir='runs/hub/exp'):
    #     save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
    #     Path(save_dir).mkdir(parents=True, exist_ok=True)
    #     self.display(save=True, save_dir=save_dir)  # save results
    #
    # def render(self):
    #     self.display(render=True)  # render results
    #     return self.imgs

    # def pandas(self):
    #     # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
    #     new = copy(self)  # return copy
    #     ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
    #     cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
    #     for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
    #         a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
    #         setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
    #     return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n
