# -*-coding:utf-8-*-
import argparse
import logging
import yaml
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict
from models import *

from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr



from torch.nn import _reduction as _Reduction
import warnings


parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()



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
        err = cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        err = 10 * torch.atan(err) + err
        return err



def train(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)

    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


def main():
    global args, config, last_epoch, best_prec, writer
    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)
    writer = SummaryWriter(log_dir=args.work_path + '/event/'+config.ckpt_name+'logexp80')



    # define netowrk
    net = get_model(config)
    #logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # define loss and optimizer
    criterion = CrossEntropyLoss_act()
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        writer.add_scalar('learning_rate', lr, epoch)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (
                epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    writer.close()
    logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))


if __name__ == "__main__":
    main()
