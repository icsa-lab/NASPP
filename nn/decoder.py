""" Decoders"""

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_NAMES = [
    'conv1','conv3','sep_conv3','sep_conv5','gap','conv3_dil3','conv3_dil12','sep_conv3_dil3','sep_conv5_dil6','skip_connect',
    'none'
]

def conv3x3(in_planes, out_planes, stride=1, bias=False, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,bias=False),
        nn.BatchNorm2d(C_out, affine=affine)
    )

def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    )

def conv_bn_relu6(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn_relu6(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inp * expand_ratio,inp * expand_ratio,3,stride,1,groups=inp * expand_ratio,bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class GAPConv1x1(nn.Module):
    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        out = nn.functional.interpolate(
            out, size=size, mode='bilinear', align_corners=False)
        return out

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(
            self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        super(SepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False))
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx), basic_op())

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        return self.bn(torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1))

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'gap': lambda C, stride, affine, repeats=1: GAPConv1x1(C, C),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),

    'conv1': lambda C, stride, affine: nn.Sequential(conv1x1(C, C, stride=stride), nn.BatchNorm2d(C, affine=affine),
                                                     nn.ReLU(inplace=False)),
    'conv3': lambda C, stride, affine: nn.Sequential(conv3x3(C, C, stride=stride), nn.BatchNorm2d(C, affine=affine),
                                                     nn.ReLU(inplace=False)),

    'sep_conv3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),

    'dil_conv3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),

    'conv3_dil3': lambda C, stride, affine: nn.Sequential(conv3x3(C, C, stride=stride, dilation=3),nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3_dil12': lambda C, stride, affine: nn.Sequential(conv3x3(C, C, stride=stride, dilation=12),
        nn.BatchNorm2d(C, affine=affine),nn.ReLU(inplace=False)),

    'sep_conv3_dil3': lambda C, stride, affine: SepConv(C, C, 3, stride, 3, affine=affine, dilation=3),
    'sep_conv5_dil6': lambda C, stride, affine: SepConv(C, C, 5, stride, 12, affine=affine, dilation=6)
}

class Cell(nn.Module):
    def __init__(self, config, inp):
        super(Cell, self).__init__()
        self._ops = nn.ModuleList()
        self._pos = []
        self._collect_inds = [0]
        self._pools = ['x']
        for ind, op in enumerate(config):
            if ind == 0:
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = OP_NAMES[op_id]
                self._ops.append(
                    OPS[op_name](inp, 1, True))
                self._pos.append(pos)
                self._collect_inds.append(ind+1)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                for (pos, op_id) in zip([pos1, pos2], [op_id1, op_id2]):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = OP_NAMES[op_id]
                    self._ops.append(OPS[op_name](inp, 1, True))
                    self._pos.append(pos)
                    self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                op_name = 'sum'
                self._ops.append(
                    FuseFeature(size_1=None, size_2=None, fused_feature_size=inp, pre_transform=False))
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append(
                    '{}({},{})'.format(op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]))

    def forward(self, x):
        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                feats.append(op(feats[pos[0]], feats[pos[1]]))
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out

class FuseFeature(nn.Module):
    def __init__(self, size_1, size_2, fused_feature_size, pre_transform=True):
        super(FuseFeature, self).__init__()
        self.pre_transform = pre_transform
        if self.pre_transform:
            self.branch_1 = conv_bn_relu(size_1, fused_feature_size, 1, 1, 0)
            self.branch_2 = conv_bn_relu(size_2, fused_feature_size, 1, 1, 0)

    def forward(self, x1, x2):
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear')(x1)
        return x1 + x2

class MNode(nn.Module):
    def __init__(self, ctx_config, conn, inps, fused_feature_size, cell):
        super(MNode, self).__init__()
        self.index_1, self.index_2 = conn
        inp_1, inp_2 = inps
        self.op_1 = cell(ctx_config, inp_1)
        self.op_2 = cell(ctx_config, inp_2)
        self.agg = FuseFeature(inp_1, inp_2, fused_feature_size)

    def forward(self, x1, x2):
        return self.agg(self.op_1(x1), self.op_2(x2))

class Decoder(nn.Module):
    def __init__(self,inp_sizes,num_classes,config,fused_feature_size,num_pools=4):
        super(Decoder, self).__init__()
        cells = []
        self.collect_inds = []
        self.pool = ['l{}'.format(i + 1) for i in range(num_pools)]
        self.info = []
        self.fused_feature_size = fused_feature_size

        for out_idx, size in enumerate(inp_sizes):
            setattr(self,'adapt{}'.format(out_idx + 1),conv_bn_relu(size, fused_feature_size, 1, 1, 0, affine=True))
            inp_sizes[out_idx] = fused_feature_size
        inp_sizes = inp_sizes.copy()
        micro_config, macro_config = config
        self.conns = macro_config
        self.collect_inds = []
        for node_idx, conn_index in enumerate(macro_config):
            for ind in conn_index:
                if ind in self.collect_inds:
                    self.collect_inds.remove(ind)
            ind_1, ind_2 = conn_index
            cells.append(MNode(micro_config, conn_index,
                                   (inp_sizes[ind_1], inp_sizes[ind_2]),
                                   fused_feature_size,
                                   Cell))
            self.collect_inds.append(node_idx + num_pools)
            inp_sizes.append(fused_feature_size)
            self.pool.append('({} + {})'.format(self.pool[ind_1], self.pool[ind_2]))
        self.cells = nn.ModuleList(cells)
        self.pre_clf = conv_bn_relu(fused_feature_size * len(self.collect_inds),
                                    fused_feature_size, 1, 1, 0)
        self.conv_clf = conv3x3(fused_feature_size, num_classes, stride=1, bias=True)
        self.info = ' + '.join(self.pool[i] for i in self.collect_inds)
        self.num_classes = num_classes

    def forward(self, x):
        x = list(x)
        for out_idx in range(len(x)):
            x[out_idx] = getattr(self, 'adapt{}'.format(out_idx + 1))(x[out_idx])
        for cell, conn_index in zip(self.cells, self.conns):
            cell_out = cell(x[conn_index[0]], x[conn_index[1]])
            x.append(cell_out)
        out = x[self.collect_inds[0]]
        for i in range(1, len(self.collect_inds)):
            collect = x[self.collect_inds[i]]
            if out.size()[2] > collect.size()[2]:
                collect = nn.Upsample(
                    size=out.size()[2:], mode='bilinear', align_corners=False)(collect)
            elif collect.size()[2] > out.size()[2]:
                out = nn.Upsample(
                    size=collect.size()[2:], mode='bilinear', align_corners=False)(out)
            out = torch.cat([out, collect], 1)

        out = self.conv_clf(self.pre_clf(F.relu(out)))
        return out
