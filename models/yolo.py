import math
import yaml
import logging
import traceback
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn

from models.common import Conv, SPPCSPC, Concat, ImplicitA, ImplicitM, RepConv, NMS, IndexDomains, \
    FeatureSum, FeatureMul, ScharrEdge, FusePol, MPM3, MPM4, CDDS3F
from utils.general import make_divisible
from utils.autoanchor import check_anchor_order
from utils.torch_utils import initialize_weights, scale_img, fuse_conv_and_bn, model_info

logger = logging.getLogger(__name__)


class IDetect(nn.Module):
    stride = None  # stride computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc, anchors=(), ch=()):
        # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of output per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', anchors)  # shape(num_layers, num_anchors, 2)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(num_layers,1,num_anchors,1,1,2)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1),
                                           4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32, device=z.device)
        box @= convert_matrix
        return (box, score)


class Model(nn.Module):
    def __init__(self, cfg='carnet_v1.yaml', input_channel=3, nc=None, anchors=None):
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, 'r') as file:
                self.yaml = yaml.load(file, Loader=yaml.SafeLoader)

        input_channel = self.yaml['input_channels'] = self.yaml.get('input_channels', input_channel)
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding models.yaml nc={} with nc={}'.format(
                self.yaml['nc'], nc))
            self.yaml['nc'] = nc
        if anchors:
            logger.info('Overriding models.yaml anchors with anchors={}'.format(anchors))
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), in_channel=[input_channel])  # models, save_list
        self.names = [str(i) for i in range(self.yaml['nc'])]

        # build stride, anchors
        m = self.model[-1]
        if isinstance(m, IDetect):
            s = 512  # 2x min stride, default 256
            # forward torch.zeros(1, input_channel, s, s)
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward([torch.zeros(1, input_channel, s, s)] * 3)])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # initialize_weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x[0].shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi, xj, xk = x[0], x[1], x[2]  # rgb, aop, dop
                # xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                xi = scale_img(xi.flip(fi) if fi else xi, si, gs=int(self.stride.max()))
                xj = scale_img(xj.flip(fi) if fi else xj, si, gs=int(self.stride.max()))
                xj = 1.0 - xj
                xk = scale_img(xk.flip(fi) if fi else xk, si, gs=int(self.stride.max()))
                x = [xi, xj, xk]
                yi = self.forward_once(x)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        # feature_maps = OrderedDict()
        for m in self.model:
            if m.i in [0, 1, 2]:  # the first three layers just used for splitting input
                xi = m(x)
                y.append(xi)
                continue  # not overwrite x's value

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, IDetect):
                    break


            try:
                x = m(x)  # run
            except RuntimeError as e:
                print(traceback.format_exc())
                # print('RuntimeError:\n{}'.format(e))
                print('m-{}: {}, from {}, input {}'.format(
                    m.i, m.t, m.f, x.shape if isinstance(x, torch.Tensor) else [item.shape for item in x ]
                ))
                exit(0)
            except ValueError as e:
                print(traceback.format_exc())
                # print('ValueError:\n{}'.format(e))
                print('m-{}: {}, from {}, input {}'.format(
                    m.i, m.t, m.f, x.shape if isinstance(x, torch.Tensor) else [item.shape for item in x]
                ))
                exit(0)
            except IndexError as e:
                print(traceback.format_exc())
                # print('IndexError:\n{}'.format(e))
                print('m-{}: {}, from {}, input {}'.format(
                    m.i, m.t, m.f, x.shape if isinstance(x, torch.Tensor) else [item.shape for item in x]
                ))
                exit(0)
            except TypeError as e:
                print(traceback.format_exc())
                # print('TypeError:\n{}'.format(e))
                print('m-{}: {}, from {}, input {}'.format(
                    m.i, m.t, m.f, x.shape if isinstance(x, torch.Tensor) else [item.shape for item in x]
                ))
                exit(0)

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))

        # if feature_maps and len(feature_maps) != 0:
        #     torch.save(feature_maps, 'runs/test/v1_feature_maps.pth')

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0, 1, 2, bc + 3)].data
            obj_idx = 2 * bc + 4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx + 1):].data += math.log(
                0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b[:, (0, 1, 2, bc + 3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse models Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect,)):
                m.fuse()
                m.forward = m.fuseforward
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].idx + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def info(self, verbose=False, img_size=640):  # print models information
        model_info(self, verbose, img_size)


def parse_model(model_cfg_dict, in_channel):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc = model_cfg_dict['anchors'], model_cfg_dict['nc']
    depth_factor, width_factor = model_cfg_dict['depth_multiple'], model_cfg_dict['width_multiple']
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_output = num_anchors * (nc + 5)

    layers, save_list, c2 = [], [], in_channel[-1]
    for i, (f, n, m, args) in enumerate(model_cfg_dict['backbone'] + model_cfg_dict['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * depth_factor), 1) if n > 1 else n  # generate depth of network
        if m in [nn.Conv2d, Conv, RepConv, SPPCSPC]:
            c1, c2 = in_channel[f], args[0]
            if c2 != num_output:
                ''' makesure out channel divisible when divided by the second argument, 
                    if current output channel is not output of the network '''
                c2 = make_divisible(c2 * width_factor, 8)
            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # repeat n times
                n = 1
        elif m is nn.BatchNorm2d:
            args = [in_channel[f]]
        elif m is Concat:
            c2 = sum([in_channel[x] for x in f])
        elif m in [IDetect]:
            args.append([in_channel[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in [IndexDomains]:
            c2 = 3
        elif m in [ScharrEdge]:
            c2 = 1
        elif m in [FeatureSum, FeatureMul]:
            c2 = in_channel[f[0]]
        elif m in [MPM3, MPM4]:
            c1 = c2 = in_channel[f]
            args = [c1]
        elif m in [FusePol, CDDS3F]:
            c1 = c2 = in_channel[f[0]]
            args = [c1]
        else:
            c2 = in_channel[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        m_type = str(m)[8:-2].replace('__main__.', '')  # module type
        num_params = sum([x.numel() for x in m_.parameters()])  # number parameters
        m_.i, m_.f, m_.t, m_.nparams = i, f, m_type, num_params  # attach index, 'from' index, type, num params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, num_params, m_type, args))  # print
        save_list.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to save list
        layers.append(m_)
        if i == 0: in_channel = []
        in_channel.append(c2)
    return nn.Sequential(*layers), sorted(save_list)


if __name__ == '__main__':
    pass
