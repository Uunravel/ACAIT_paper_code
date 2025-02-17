"""Utility functions to construct a model."""

import torch
import numpy as np
from torch import nn

from extensions import data_parallel
from extensions import model_refinery_wrapper
from extensions import refinery_loss
from models.efficientnet_EficientNetB0_C10_deformed import efficientnet_b0, efficientnet_b3, Margloss, efficientnet_b1, efficientnet_b2, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, CrossEntropyLoss_act
from models.efficientnet_ex import efficientnet_ex, efficientnet_exx

MODEL_NAME_MAP = {
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_ex': efficientnet_ex,
    'efficientnet_exx': efficientnet_exx,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
}

def _get_softmax_dim(name, ndim, stacklevel):
    # type: (str, int, int) -> int
    warnings.warn("Implicit dimension choice for {} has been deprecated. "
                  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret

#
#def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
#    # type: (Tensor,     Optional[int], int, Optional[int]) -> Tensor
#    if dim is None:
#        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
#
#    if dtype is None:
#
#        ret = input.log_softmax(dim)
#
#    else:
#
#        ret = input.log_softmax(dim, dtype=dtype)
#
#    return ret
#
#
#def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
#
#             reduce=None, reduction='mean'):
#    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
#
#    if size_average is not None or reduce is not None:
#        reduction = _Reduction.legacy_get_string(size_average, reduce)
#
#    dim = input.dim()
#
#    if dim < 2:
#        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))
#
#    if input.size(0) != target.size(0):
#        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
#
#                         .format(input.size(0), target.size(0)))
#
#    if dim == 2:
#
#        ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
#
#    elif dim == 4:
#
#        ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
#
#    else:
#
#        # dim == 3 or dim > 4
#
#        n = input.size(0)
#
#        c = input.size(1)
#
#        out_size = (n,) + input.size()[2:]
#
#        if target.size()[1:] != input.size()[2:]:
#            raise ValueError('Expected target size {}, got {}'.format(
#
#                out_size, target.size()))
#
#        input = input.contiguous().view(n, c, 1, -1)
#
#        target = target.contiguous().view(n, 1, -1)
#
#        reduction_enum = _Reduction.get_enum(reduction)
#
#        if reduction != 'none':
#
#            ret = torch._C._nn.nll_loss2d(
#
#                input, target, weight, reduction_enum, ignore_index)
#
#        else:
#
#            out = torch._C._nn.nll_loss2d(
#
#                input, target, weight, reduction_enum, ignore_index)
#
#            ret = out.view(out_size)
#
#    return ret
#
#
#def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
#
#                  reduce=None, reduction='mean'):
#    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
#
#    if size_average is not None or reduce is not None:
#        reduction = _Reduction.legacy_get_string(size_average, reduce)
#
#    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
#
#
#class _Loss(nn.Module):
#
#    def __init__(self, size_average=None, reduce=None, reduction='mean'):
#
#        super(_Loss, self).__init__()
#
#        if size_average is not None or reduce is not None:
#
#            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
#
#        else:
#
#            self.reduction = reduction
#
#
#class _WeightedLoss(_Loss):
#
#    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
#        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
#
#        self.register_buffer('weight', weight)
#
#
#class CrossEntropyLoss_act(_WeightedLoss):
#    __constants__ = ['weight', 'ignore_index', 'reduction']
#
#    def __init__(self, weight=None, size_average=None, ignore_index=-100,
#
#                 reduce=None, reduction='mean'):
#        super(CrossEntropyLoss_act, self).__init__(weight, size_average, reduce, reduction)
#
#        self.ignore_index = ignore_index
#
#        self.err = 0
#
#    def forward(self, input, target):
#        # F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
#
#        err = cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
#        # print('err_ori',err)
#        err = 18.0/np.pi*torch.atan(err) ** 2 + err
#
#        # err = torch.atan(2*err) ** 3 + torch.atan(0.3*err) ** 3 + torch.log(err + 1)
#        # err = -0.05 * torch.exp(-15.0 * (err) + 0.5) + (0.05 * (err)) ** 2
#        # err = (torch.exp(err)-0.95)
#        # err = 3*torch.log(torch.exp(err)-1)
#        # err = (2.5*torch.atan(3*err))**2 + err
#
#        # err = (torch.atan(20*err))**4 + err
#        # err = 5 * torch.atan(err) + err
#        # err = 5*err
#        # return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
#        # print('err:',err)
#        return err
        
def _create_single_cpu_model(model_name, state_file=None, cosln=False):
    if model_name not in MODEL_NAME_MAP:
        raise ValueError("Model {} is invalid. Pick from {}.".format(
            model_name, sorted(MODEL_NAME_MAP.keys())))
    model_class = MODEL_NAME_MAP[model_name]
    model = model_class(num_classes=100, coslinear=cosln)
    if state_file is not None:
        model.load_state_dict(torch.load(state_file))
    return model


def create_model(model_name, model_state_file=None, gpus=[0,1,2,3], label_refinery=None,
                 label_refinery_state_file=None, coslinear=True, scale=5.0):
    model = _create_single_cpu_model(model_name, model_state_file, coslinear)
    if label_refinery is not None:
        assert label_refinery_state_file is not None, "Refinery state is None."
        label_refinery = _create_single_cpu_model(
            label_refinery, label_refinery_state_file, coslinear)
        model = model_refinery_wrapper.ModelRefineryWrapper(model, label_refinery, scale)
        loss = refinery_loss.RefineryLoss(cosln=coslinear, scl=scale)
    else:
        if coslinear:
            print('Using other loss')
            loss = Margloss(s=scale)
        else:
            print('Using CrossEntropyLoss')
            # loss = F.cross_entropy
            loss = CrossEntropyLoss_act()

    if len(gpus) > 0:
        model = model.cuda()
        loss = loss.cuda()
    # if len(gpus) > 1:
        # model = data_parallel.DataParallel(model, device_ids=gpus)
    model = data_parallel.DataParallel(model)
    return model, loss
