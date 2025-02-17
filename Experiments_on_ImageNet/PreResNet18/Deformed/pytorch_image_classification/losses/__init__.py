from typing import Callable, Tuple

import torch.nn as nn
import yacs.config

from .cutmix import CutMixLoss
from .mixup import MixupLoss
from .ricap import RICAPLoss
from .dual_cutout import DualCutoutLoss
from .label_smoothing import LabelSmoothingLoss
from torch.nn import _reduction as _Reduction
import torch
def _get_softmax_dim(name, ndim, stacklevel):
    # type: (str, int, int) -> int
    warnings.warn("Implicit dimension choice for {} has been deprecated. "
                  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    # type: (Tensor,     Optional[int], int, Optional[int]) -> Tensor
    if dim is None:
        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)

    if dtype is None:

        ret = input.log_softmax(dim)

    else:

        ret = input.log_softmax(dim, dtype=dtype)

    return ret


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,

             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()

    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'

                         .format(input.size(0), target.size(0)))

    if dim == 2:

        ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

    elif dim == 4:

        ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

    else:

        # dim == 3 or dim > 4

        n = input.size(0)

        c = input.size(1)

        out_size = (n,) + input.size()[2:]

        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(

                out_size, target.size()))

        input = input.contiguous().view(n, c, 1, -1)

        target = target.contiguous().view(n, 1, -1)

        reduction_enum = _Reduction.get_enum(reduction)

        if reduction != 'none':

            ret = torch._C._nn.nll_loss2d(

                input, target, weight, reduction_enum, ignore_index)

        else:

            out = torch._C._nn.nll_loss2d(

                input, target, weight, reduction_enum, ignore_index)

            ret = out.view(out_size)

    return ret


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,

                  reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)


class _Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):

        super(_Loss, self).__init__()

        if size_average is not None or reduce is not None:

            self.reduction = _Reduction.legacy_get_string(size_average, reduce)

        else:

            self.reduction = reduction


class _WeightedLoss(_Loss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)

        self.register_buffer('weight', weight)


class CrossEntropyLoss_act(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,

                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss_act, self).__init__(weight, size_average, reduce, reduction)

        self.ignore_index = ignore_index

        self.err = 0

    def forward(self, input, target):
        # F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        err = cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        err = torch.log(torch.exp(err)-1.9)
        # return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        return err
        
def create_loss(config: yacs.config.CfgNode) -> Tuple[Callable, Callable]:
    if config.augmentation.use_mixup:
        train_loss = MixupLoss(reduction='mean')
    elif config.augmentation.use_ricap:
        train_loss = RICAPLoss(reduction='mean')
    elif config.augmentation.use_cutmix:
        train_loss = CutMixLoss(reduction='mean')
    elif config.augmentation.use_label_smoothing:
        train_loss = LabelSmoothingLoss(config, reduction='mean')
    elif config.augmentation.use_dual_cutout:
        train_loss = DualCutoutLoss(config, reduction='mean')
    else:
        #train_loss = nn.CrossEntropyLoss(reduction='mean')
        train_loss = CrossEntropyLoss_act(reduction='mean')
    val_loss = nn.CrossEntropyLoss(reduction='mean')
    return train_loss, val_loss
