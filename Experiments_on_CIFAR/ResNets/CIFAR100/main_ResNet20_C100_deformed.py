from __future__ import print_function
import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel

import model_loader
import dataloader

import argparse

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict
from models import *

# from utils import Logger, count_parameters, data_augmentation, \
#     load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
#     save_checkpoint, adjust_learning_rate, get_current_lr
from torch.optim.optimizer import Optimizer, required



from torch.nn import _reduction as _Reduction
import warnings
from collections.abc import Iterable

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
        err = 18 / np.pi*(torch.atan(err))**2 + err
        return err


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training
def train(trainloader, net, criterion, optimizer, epoch, use_cuda=True):
    global writer

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if isinstance(criterion, CrossEntropyLoss_act):
    # if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 100).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()
    writer.add_scalar('train_loss', train_loss/total, global_step=epoch)
    writer.add_scalar('train_acc', 100.*correct/total, global_step=epoch)
    return train_loss/total, 100.*correct/total


def test(testloader, net, criterion, epoch, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    global best_prec, writer

    if isinstance(criterion, CrossEntropyLoss_act):
    # if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 100).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()
    writer.add_scalar('test_loss', test_loss/total, global_step=epoch)
    writer.add_scalar('test_acc', 100.*correct/total, global_step=epoch)
    return test_loss/total, 100.*correct/total

def name_save_folder(args):
    save_folder = args.model + '_pi3atan2_' + str(args.optimizer) + '_lr=' + str(args.lr)
    if args.lr_decay != 0.1:
        save_folder += '_lr_decay=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mom=' + str(args.momentum)
    save_folder += '_save_epoch=' + str(args.save_epoch)
    if args.loss_name != 'crossentropy':
        save_folder += '_loss=' + str(args.loss_name)
    if args.noaug:
        save_folder += '_noaug'
    if args.raw_data:
        save_folder += '_rawdata'
    if args.label_corrupt_prob > 0:
        save_folder += '_randlabel=' + str(args.label_corrupt_prob)
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)

    return save_folder


class SGD_loss(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_loss, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_loss, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, train_loss, epoch, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                swich = 0
                # if_print = 0
                # if epoch%1==0:
                    # print('d_p_ori:', d_p.max().max(),d_p.min().min())
                    # print('p_ori:',p.max().max(),p.min().min())
                # if_print
                # d_p = 1 / (1 + np.exp(-(14 * train_loss - 13))) * d_p
                train_loss = -1
                if train_loss != -1:
                    d_p = (2.0 / (1 + np.exp(-2*train_loss)) - 1.0)*d_p
                if epoch<0:  #5 is optimal
                    d_p = d_p.sign()*0.2*torch.exp(-100000*d_p.abs())
                    swich = 1
                    # print('d_p_rev:',d_p)
                # d_p = torch.tan(1.3*d_p)
                else:

                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        elif swich !=1:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                            if train_loss != -1:
                                buf.mul_((2.0 / (1 + np.exp(-(2*train_loss))) - 1.0))

                            # buf.mul_(1 / (1 + np.exp(-(14 * train_loss - 13))))
                        # if train_loss != -1:
                            # print('d_p:', d_p)
                            if nesterov:
                                d_p = d_p.add(momentum, buf)
                            else:
                                d_p = buf
                        else:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            d_p = buf
                swich = 0

                # direct_loss = d_p
                # if train_loss != -1:
                # if epoch%3==0 :  #5 is optimal
                    # print('nesterov:',d_p)
                    # direct_loss = direct_loss.sign()*0.1*torch.exp(-10000*direct_loss.abs())#0.1:-100

                    # d_p = d_p.sign()*0.05*torch.exp(-10000*d_p.abs())
                    # print('direct_loss:', direct_loss)
                    # print('p:', p)
                    # print('direct_loss', direct_loss)
                #     train_loss_temp =train_loss

                #     if train_loss < 5.0:
                #         train_loss_temp = 0.2 * train_loss
                #     elif (train_loss >= 5.0) & (train_loss < 10.0):
                #         train_loss_temp = 0.5 * (train_loss - 3.0)
                #     else:
                #         train_loss_temp = train_loss - 8.0

                #     # if train_loss < 9.0:
                #     #     train_loss_temp = 0.1 * train_loss
                #     # elif (train_loss >= 9.0) & (train_loss < 19.0):
                #     #     train_loss_temp = 0.4 * (train_loss - 8.0)
                #     # else:
                #     #     train_loss_temp = train_loss - 15.0
                #     direct_loss = (2.0 / np.pi * np.arctan(train_loss_temp))*direct_loss
                #     buf.mul_(2.0 / np.pi * np.arctan(train_loss_temp))

                    # print('train_loss in epoch:', train_loss, epoch)

                    # buf = (2.0 / np.pi * np.arctan(train_loss_temp))*buf
                    # if epoch != 0:
                        # print('d_p:\n', d_p.abs().max().max(), d_p.abs().min().min(), d_p)
                        # print('p', p)`
                        # print('train_loss**2',train_loss**2)
                        # train_loss = 0.5*(7/np.pi*np.arctan(train_loss))**2
                        # direct_loss = train_loss*d_p

                        # train_loss = np.log(5*train_loss+1)
                        # print('train_loss:\n', train_loss)
                        # direct_loss = 0.4*d_p.sign()*train_loss/(10000*d_p.abs()+3)
                        # direct_loss = d_p*torch.log(torch.tensor(train_loss).cuda())
                        # direct_loss = 0.002*d_p*train_loss

                        # direct_loss = p.abs().mul(torch.sign(d_p)*train_loss)
                        # direct_loss = 1.0/1.507*torch.atan(direct_loss)

                        # direct_loss = train_loss**2*((-50*d_p).abs()).exp()*0.000002*d_p.sign()


                        # direct_loss = train_loss**2/(d_p.abs()+1)*0.0002*d_p.sign()
                        # print('direct_loss2:\n', direct_loss.abs().max().max(),direct_loss.abs().min().min())
                        # direct_loss = 0.00001*torch.atan(torch.tensor(100000.0)*train_loss)*torch.sign(d_p)
                        # if epoch == 2:
                            # print('d_p:\n', d_p)
                            # print('direct_loss:\n', direct_loss)

                # else:
                #     direct_loss = d_p
                # if epoch !=0 & epoch % 10 == 0:
                #     p.data.add_(-0.01 * group['lr'], direct_loss)
                # elif epoch !=0 & epoch % 5 == 0:
                #     p.data.add_(-0.1 * group['lr'], direct_loss)
                # elif epoch !=0 & epoch % 2 == 0:
                #     p.data.add_(-5 * group['lr'], direct_loss)
                # else:
                #     p.data.add_(-group['lr'], direct_loss)

                # if epoch%20<5 :  #5 is optimal
                #     d_p = d_p.sign()*0.02*torch.exp(-10000*d_p.abs())
                #
                # # print(d_p)
                # # d_p = torch.tan(1.3*d_p)
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)
                # if (epoch%20>=5):
                # # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                # # if train_loss != -1:
                #     # print('d_p:', d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf

                # print('d_p_final:', d_p)
                #
                # times = epoch-278
                # d_p = 1/(1+np.exp(-(14*train_loss-13))) * d_p
                # p.data.add_(-0.1, d_p)
                # p.data.add_(-0.0000001, d_p)

        # print(-0.00000001 * 10 ** times)
                p.data.add_(-group['lr'], d_p)
        return loss


if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=10, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=3, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # model parameters
    parser.add_argument('--model', '-m', default='resnet20')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')

    args = parser.parse_args()

    print('\nLearning Rate: %f' % args.lr)
    print('\nDecay Rate: %f' % args.lr_decay)

    use_cuda = torch.cuda.is_available()
    print('Current devices: ' + str(torch.cuda.current_device()))
    print('Device count: ' + str(torch.cuda.device_count()))

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    lr = args.lr  # current learning rate
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    save_folder = name_save_folder(args)
    if not os.path.exists('trained_nets/' + save_folder):
        os.makedirs('trained_nets/' + save_folder)

    f = open('trained_nets/' + save_folder + '/log.out', 'a', 1)

    trainloader, testloader = dataloader.get_data_loaders(args)

    if args.label_corrupt_prob and not args.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args.model)
        # print(net)
        init_params(net)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    criterion = CrossEntropyLoss_act()
    # criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_opt:
        checkpoint_opt = torch.load(args.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])
    global config, last_epoch, best_prec, writer
    writer = SummaryWriter('./event')
    # record the performance of initial model
    if not args.resume_model:
        train_loss, train_err = test(trainloader, net, criterion, use_cuda)
        test_loss, test_err = test(testloader, net, criterion, use_cuda)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (0, train_loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        state = {
            'acc': 100 - test_err,
            'epoch': 0,
            'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'trained_nets/' + save_folder + '/model_0.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_0.t7')

    for epoch in range(start_epoch, args.epochs + 1):

        loss, train_err = train(trainloader, net, criterion, optimizer, epoch, use_cuda)
        test_loss, test_err = test(testloader, net, criterion, epoch, use_cuda)

        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        # Save checkpoint.
        acc = 100 - test_err
        if epoch == 1 or epoch % args.save_epoch == 0 or epoch == 150:
            state = {
                'acc': acc,
                'epoch': epoch,
                'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict(),
            }
            opt_state = {
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
            torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')

        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay
    writer.close()
    f.close()
