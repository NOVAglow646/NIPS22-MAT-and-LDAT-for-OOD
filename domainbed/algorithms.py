# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from telnetlib import PRAGMA_HEARTBEAT
from tensorboardX.summary import hparams
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import collections
import copy
import numpy as np
from collections import defaultdict

from torch.optim import optimizer
import os

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, ParamDict
from torchvision import utils as vutils
from torchvision import transforms

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',  # SAND-mask
    'IGA',
    'SelfReg',
    'AT',
    'UAT',
    'MAT',
    'LDAT',
    'LAT'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


def load_model(model_path, algo):
    model_dict = torch.load(model_path)
    algorithm_class = get_algorithm_class(algo)
    #print(model_dict['model_hparams'])#(2,28,28)
    algorithm = algorithm_class(model_dict['model_input_shape'], model_dict['model_num_classes'],
                                model_dict['model_num_domains'], model_dict['model_hparams'])
    algorithm.load_state_dict(model_dict['model_dict'])
    return algorithm


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes,
                                              self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        # self.network = networks.Lin(input_shape, num_classes)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.hparams["lr"],
                                          weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class MAT(ERM):
    """
    Multi-perturbation Adversarial training
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MAT, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.delta = None
        self.register_buffer('update_count', torch.tensor([0]))

        if hparams['is_cmnist'] == 0:  # for non-CMNIST datasets
            self.std = torch.Tensor([0.229, 0.224,
                                     0.225]).view(3, 1, 1).unsqueeze(0).expand(hparams['batch_size'],
                                                                               *input_shape).cuda()
            self.mean = torch.Tensor([0.485, 0.456,
                                      0.406]).view(3, 1, 1).unsqueeze(0).expand(hparams['batch_size'],
                                                                                *input_shape).cuda()
            self.high = torch.Tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]).cuda()
            self.low = torch.Tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).cuda()

        else:
            self.high = torch.Tensor([1, 1]).cuda()
            self.low = torch.Tensor([0, 0]).cuda()
            self.std = 1
            self.mean = 0

    def k_projection(self, x, eps):
        '''Given a tensor x with shape [k, c, h, w], project the last 3 channels x into a ball with radius eps'''
        nm = torch.linalg.norm(x, dim=(1, 2, 3))
        nm.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        ones = torch.ones_like(nm)
        r = (eps / (nm + 1e-10))
        return x * torch.min(ones, r)

    def projection(self, x, eps):
        '''Project x into a ball with radius eps'''
        return x * min(1, eps / (torch.linalg.norm(x) + 1e-10))

    def clamp(self, x):  
        '''Crop x to [self.low, self.high]'''
        x = x.permute(0, 2, 3, 1)
        x = torch.max(torch.min(x, self.high), self.low)
        x = x.permute(0, 3, 1, 2)
        return x

    def make_alpha_delta(self, x_shape, alpha, d):
        '''Given alpha and k perturbations (d), return linear combined perturbation'''
        a_delta = torch.zeros(x_shape).cuda()
        #bsz = x_shape[0]
        d = d.unsqueeze(0)  #BKCHW
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  #BKCHW
        a_delta = torch.sum(alpha * d, dim=1)
        return a_delta

    def FGSM(self, x, y, alpha, d, num_iter=1, norm=2):
        '''Inner Fast Gradient Sign Attack maximization'''
        epsilon = self.hparams["at_eps"]
        x_denorm = x * self.std + self.mean
        for t in range(num_iter):
            alpha_softmax = torch.softmax(alpha, dim=1)
            a_delta = self.make_alpha_delta(x.shape, alpha_softmax, d)
            loss = F.cross_entropy(self.predict(self.clamp((x_denorm + a_delta - self.mean) / self.std)),
                                   y,
                                   reduction='none')
            loss = torch.mean(loss.clamp(0, 2))
            (a_grad, d_grad) = autograd.grad(loss, [alpha, d], retain_graph=False)
            # l2 norm
            if norm == 2:
                d_grad = self.hparams["at_alpha"] * d_grad.detach() / (torch.linalg.norm(d_grad) + 1e-10
                                                                       )  
                d = d + d_grad # Gradient Acsent
                d = self.k_projection(d, epsilon)
                alpha = alpha + self.hparams["kat_alpha_step"] * a_grad
            # l_inf norm
            elif norm == 'inf':
                grad = self.hparams["at_alpha"] * grad.detach().sign()
                d = (d + grad).clamp(min=-epsilon, max=epsilon)
            else:
                raise NotImplementedError
        return alpha.detach(), d.detach()

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        '''Conduct one epoch of optimization'''
        bsz = len(minibatches[0][0])
        env_num = len(minibatches)
        if self.delta == None:
            randomize = True
            if randomize:
                self.delta = torch.randn((env_num, int(self.hparams["at_delta_num"]), *minibatches[0][0][0].shape),
                                         requires_grad=True).cuda()
                self.delta.data = self.delta.data * 2 * \
                    self.hparams["at_eps"] - self.hparams["at_eps"]
            else:
                self.delta = torch.zeros(
                    (env_num, int(self.hparams["at_delta_num"]), *minibatches[0][0][0].shape)).cuda()

        alpha = torch.randn(  
            (env_num, bsz, int(self.hparams["at_delta_num"])), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        loss = 0.
        for i, (x, y) in enumerate(minibatches):  # For every domain i, update alpha and delta. (x,y) is a batch.
            alpha[i], self.delta[i] = self.FGSM(x,
                                                y,
                                                alpha[i],
                                                self.delta[i],
                                                num_iter=self.hparams['KAT_num_iter'],
                                                norm=2)
            a_delta = self.make_alpha_delta(x.shape, torch.softmax(alpha[i], dim=1), self.delta[i])
            x_denorm = x * self.std + self.mean
            x_adv = self.clamp((x_denorm + a_delta - self.mean) / self.std).detach()
            loss_adv = F.cross_entropy(self.predict(x_adv), y)
            loss += loss_adv.item()
            loss_adv.backward()
        self.optimizer.step()
        del loss_adv
        self.update_count += 1
        return {'loss': loss}


class LDAT(ERM):
    '''Low-rank Decomposed Adversarial training'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LDAT, self).__init__(input_shape, num_classes, num_domains, hparams)
        #self.delta = None
        self.B = None
        self.A = None
        self.register_buffer('update_count', torch.tensor([0]))

        if hparams['is_cmnist'] == 0:
            self.std = torch.Tensor([0.229, 0.224,
                                     0.225]).view(3, 1, 1).unsqueeze(0).expand(hparams['batch_size'],
                                                                               *input_shape).cuda()
            #print(self.std.shape)
            self.mean = torch.Tensor([0.485, 0.456,
                                      0.406]).view(3, 1, 1).unsqueeze(0).expand(hparams['batch_size'],
                                                                                *input_shape).cuda()
            self.high = torch.Tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]).cuda()
            self.low = torch.Tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).cuda()

        else:

            self.high = torch.Tensor([1, 1]).cuda()
            self.low = torch.Tensor([0, 0]).cuda()
            self.std = 1
            self.mean = 0

    def projection(self, x, eps):
        '''Project x into a ball with radius eps'''
        return x * min(1, eps / (torch.linalg.norm(x) + 1e-10))


    def clamp(self, x):  
        '''Crop x to [self.low, self.high]'''
        x = x.permute(0, 2, 3, 1)
        x = torch.max(torch.min(x, self.high), self.low)
        x = x.permute(0, 3, 1, 2)
        return x

    def FGSM(self, x, y, B, A, bsz, num_iter=1, norm=2):
        '''Inner Fast Gradient Sign Attack maximization'''
        delta = self.projection(torch.bmm(B, A), self.hparams['at_eps']).cuda()
        epsilon = self.hparams["at_eps"]
        x_denorm = x * self.std + self.mean
        for t in range(num_iter):
            loss = F.cross_entropy(self.predict(self.clamp((x_denorm + delta - self.mean) / self.std)),
                                   y,
                                   reduction='none')
            loss = torch.mean(loss.clamp(0, 2))  # adding beta as upperbound
            (A_grad, B_grad) = autograd.grad(loss, [A, B], retain_graph=False)
            if norm == 2:
                B_grad = self.hparams["at_alpha"] * B_grad.detach() / (torch.linalg.norm(B_grad) + 1e-10
                                                                       )  
                A_grad = self.hparams["at_alpha"] * A_grad.detach() / (torch.linalg.norm(A_grad) + 1e-10)
                B = B + B_grad * self.hparams['B_lr']
                A = A + A_grad * self.hparams['A_lr']
            elif norm == 'inf':  
                grad = self.hparams["at_alpha"] * grad.detach().sign()
                d = (d + grad).clamp(min=-epsilon, max=epsilon)
            else:
                raise NotImplementedError

        return B.detach(), A.detach()

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        '''Conduct one epoch of optimization'''
        bsz = len(minibatches[0][0])
        env_num = len(minibatches)
        channel_num = minibatches[0][0][0].shape[0]
        x_h = minibatches[0][0][0].shape[1]
        x_w = minibatches[0][0][0].shape[2]
        if self.B == None and self.A == None:
            self.B = torch.randn((env_num, channel_num, x_h, int(self.hparams["at_cb_rank"])),
                                 requires_grad=True).cuda()
            self.A = torch.randn((env_num, channel_num, int(self.hparams["at_cb_rank"]), x_w),
                                 requires_grad=True).cuda()
        delta = torch.zeros(env_num, channel_num, x_h, x_w).cuda()

        self.optimizer.zero_grad()
        loss = 0.
        for i, (x, y) in enumerate(minibatches):
            '''Update A and B for each domain i. (x,y) is batch data.'''
            self.B[i], self.A[i] = self.FGSM(x, y, self.B[i], self.A[i], bsz, norm=2)
            x_denorm = x * self.std + self.mean
            delta[i] = self.projection(torch.bmm(self.B[i], self.A[i]), self.hparams["at_eps"])  #!
            x_adv = self.clamp((x_denorm + delta[i] - self.mean) / self.std).detach()
            loss_adv = F.cross_entropy(self.predict(x_adv), y)
            loss += loss_adv.item()
            loss_adv.backward()
        self.optimizer.step()
        del loss_adv
        self.update_count += 1
        return {'loss': loss}


class AT(ERM):
    """
    Sample-wise Adversarial training
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(AT, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.high = torch.Tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]).cuda()
        self.low = torch.Tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).cuda()
        #self.high = torch.Tensor([1, 1]).cuda()
        #self.low = torch.Tensor([0, 0]).cuda()

    def projection(self, x, eps):
        return x * min(1, eps / (torch.linalg.norm(x) + 1e-10))

    def clamp(self, x):
        x = x.permute(0, 2, 3, 1)
        x = torch.max(torch.min(x, self.high), self.low)
        x = x.permute(0, 3, 1, 2)
        return x

    def FGSM(self, x, y, criterion, num_iter=10, randomize=True, norm=2):
        epsilon = self.hparams["at_eps"]
        if randomize:
            delta = torch.rand_like(x, requires_grad=True).cuda()
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(x, requires_grad=True).cuda()
        for t in range(num_iter):
            loss = criterion(self.predict(x + delta), y)
            grad = autograd.grad(loss, delta)[0]
            # l2 norm
            if norm == 2:
                grad = self.hparams["at_alpha"] * \
                    grad.detach() / (torch.linalg.norm(grad) + 1e-10)  
                delta = self.projection(delta + grad, epsilon)
            # l_inf norm
            elif norm == 'inf':
                grad = self.hparams["at_alpha"] * grad.detach().sign()
                delta = (delta + grad).clamp(-epsilon, epsilon)
            else:
                raise NotImplementedError
            delta = self.clamp(delta + x) - x
        return delta

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        self.optimizer.zero_grad()
        loss = 0.
        for i, (x, y) in enumerate(minibatches):
            x_adv = self.clamp(x + self.FGSM(x, y, F.cross_entropy, norm=2).detach())  
            loss_adv = F.cross_entropy(self.predict(x_adv), y)
            loss += loss_adv.item()
            loss_adv.backward()
        self.optimizer.step()
        return {'loss': loss}


class UAT(ERM):
    """
    Universal Adversarial training
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(UAT, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.delta = None
        self.register_buffer('update_count', torch.tensor([0]))
        self.high = torch.Tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]).cuda()
        self.low = torch.Tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).cuda()
        '''
        self.high = torch.Tensor(
            [1, 1]).cuda()
        self.low = torch.Tensor(
            [0, 0]).cuda()
        '''

    def projection(self, x, eps):
        return x * min(1, eps / (torch.linalg.norm(x) + 1e-10))

    def clamp(self, x):
        x = x.permute(0, 2, 3, 1)
        x = torch.max(torch.min(x, self.high), self.low)
        x = x.permute(0, 3, 1, 2)
        return x

    def FGSM(self, x, y, d, num_iter=1, norm=2):
        epsilon = self.hparams["at_eps"]
        for t in range(num_iter):
            loss = F.cross_entropy(self.predict(self.clamp(x + d)), y, reduction='none')
            loss = torch.sum(loss.clamp(0, 2))  
            grad = autograd.grad(loss, d)[0]
            # l2 norm
            if norm == 2:
                grad = self.hparams["at_alpha"] * \
                    grad.detach() / (torch.linalg.norm(grad) + 1e-10)  
                d = self.projection(d + grad, epsilon)
            # l_inf norm
            elif norm == 'inf':
                grad = self.hparams["at_alpha"] * grad.detach().sign()
                d = (d + grad).clamp(min=-epsilon, max=epsilon)
            else:
                raise NotImplementedError
        return d.detach()

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        if self.delta == None:
            randomize = True
            if randomize:
                self.delta = torch.randn((len(minibatches), *minibatches[0][0][0].shape), requires_grad=True).cuda()
                self.delta.data = self.delta.data * 2 * \
                    self.hparams["at_eps"] - self.hparams["at_eps"]
            else:
                self.delta = torch.zeros((len(minibatches), *minibatches[0][0][0].shape), requires_grad=True).cuda()
        self.optimizer.zero_grad()
        loss = 0.
        for i, (x, y) in enumerate(minibatches):
            self.delta[i] = self.FGSM(x, y, self.delta[i], norm=2)
            x_adv = self.clamp(x + self.delta[i]).detach()  
            loss_adv = F.cross_entropy(self.predict(x_adv), y)
            loss += loss_adv.item()
            loss_adv.backward()
        self.optimizer.step()
        del loss_adv
        self.update_count += 1
        return {'loss': loss}


class LAT(Algorithm):
    """
    Linearized Adversarial training
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LAT, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.network = networks.Net(input_shape, num_classes, hparams)
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.hparams["lr"],
                                          weight_decay=self.hparams['weight_decay'])
        self.register_buffer('update_count', torch.tensor([0]))

    def calculate_penalty(self, x, y, eps=1e-5):
        # delta = torch.zeros_like(x, requires_grad=True).cuda()
        # temp_loss = F.cross_entropy(self.predict(x + delta), y)
        x.requires_grad = True
        temp_loss = F.cross_entropy(self.predict(x), y)
        # temp_loss.backward()
        delta = torch.autograd.grad(temp_loss, x, create_graph=True)[0]
        # delta.data = delta.grad.detach()  # .sign()
        # delta.grad.zero_()
        return torch.sum(x.mul(delta))**2
        # x_adv = (1+eps)*x
        # p = self.predict(x)
        # p_adv = self.predict(x_adv)
        # prob = torch.softmax(p, dim=1)
        # pure_loss = F.cross_entropy(p, y)
        # adv_loss = F.cross_entropy(p_adv, y)
        # # attained by calculating gradient of log softmax
        # Term_1 = torch.sum((p * p) * prob)
        # Temp = p * prob
        # Term_2 = -torch.sum(torch.mm(Temp.T, Temp))
        # Second_Order_Taylor = 0.5 * (Term_1 + Term_2)
        # # removed multiplying by eps
        # one_hot = F.one_hot(y, num_classes=self.num_classes)
        # return (((adv_loss - pure_loss) / eps**2 - Second_Order_Taylor)*eps)**2, (torch.sum(torch.diag(torch.mm(p.T, prob-one_hot))))**2

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        self.optimizer.zero_grad()
        penalty_weight = (self.hparams['lat_lambda']
                          if self.update_count >= self.hparams['lat_penalty_anneal_iters'] else 1.0)
        nll = 0.
        penalty = 0.
        for x, y in minibatches:
            logits = self.predict(x)
            nll += F.cross_entropy(logits, y)
            p = self.calculate_penalty(x, y)
            penalty += p

        nll /= len(minibatches)
        penalty /= len(minibatches)
        penalty *= penalty_weight
        loss = nll + penalty

        if self.update_count == self.hparams['lat_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.hparams["lr"],
                                              weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}

    def predict(self, x):
        return self.network(x)


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.hparams["lr"],
                                          weight_decay=self.hparams['weight_decay'])
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape,
                                                self.num_classes,
                                                self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(self.network_inner.parameters(),
                                                lr=self.hparams["lr"],
                                                weight_decay=self.hparams['weight_decay'])
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(meta_weights=self.network.state_dict(),
                                 inner_weights=self.network_inner.state_dict(),
                                 lr_meta=self.hparams["meta_lr"])
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0], ) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes,
                                              self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam((list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                                        lr=self.hparams["lr_g"],
                                        weight_decay=self.hparams['weight_decay_g'],
                                        betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [torch.full((x.shape[0], ), i, dtype=torch.int64, device=device) for i, (x, y) in enumerate(minibatches)])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss + (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape,
                                   num_classes,
                                   num_domains,
                                   hparams,
                                   conditional=False,
                                   class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape,
                                    num_classes,
                                    num_domains,
                                    hparams,
                                    conditional=True,
                                    class_balance=True)


# class IRM(Algorithm):
#     """Invariant Risk Minimization"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(IRM, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         self.network = networks.Net(input_shape, num_classes, hparams)
#         self.num_classes = num_classes
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )
#         self.register_buffer('update_count', torch.tensor([0]))

#     @ staticmethod
#     def _irm_penalty(logits, y, Print=False):
#         device = "cuda" if logits[0][0].is_cuda else "cpu"
#         scale = torch.tensor(1.).to(device).requires_grad_()
#         loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
#         loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
#         grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
#         grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
#         if Print:
#             print(grad_1.norm().item(), grad_2.norm().item())
#         result = torch.sum(grad_1 * grad_2)
#         return result

#     @ staticmethod
#     def _irm_penalty2(logits, y):
#         device = "cuda" if logits[0][0].is_cuda else "cpu"
#         scale = torch.tensor(1.).to(device).requires_grad_()
#         loss_1 = F.cross_entropy(logits * scale, y)
#         grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
#         result = torch.norm(grad_1, p=1)**2
#         return result

#     def update(self, minibatches, unlabeled=None):
#         device = "cuda" if minibatches[0][0].is_cuda else "cpu"
#         penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                           >= self.hparams['irm_penalty_anneal_iters'] else
#                           1.0)
#         nll = 0.
#         penalty = 0.

#         all_x = torch.cat([x for x, y in minibatches])
#         all_logits = self.network(all_x)
#         all_logits_idx = 0
#         if self.update_count % 50 == 0:
#             Print = 1
#         else:
#             Print = 0
#         for i, (x, y) in enumerate(minibatches):
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             penalty += self._irm_penalty(logits, y, Print=Print)
#         nll /= len(minibatches)
#         penalty /= len(minibatches)
#         loss = nll + (penalty_weight * penalty)

#         if self.update_count == self.hparams['irm_penalty_anneal_iters']:
#             # Reset Adam, because it doesn't like the sharp jump in gradient
#             # magnitudes that happens at this step.
#             self.optimizer = torch.optim.Adam(
#                 self.network.parameters(),
#                 lr=self.hparams["lr"],
#                 weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.update_count += 1
#         return {'loss': loss.item(), 'nll': nll.item(),
#                 'penalty': penalty.item()}

#     def predict(self, x):
#         return self.network(x)


class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda']
                          if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.hparams["lr"],
                                              weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean)**2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.hparams["lr"],
                                              weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(inner_net.parameters(),
                                         lr=self.hparams["lr"],
                                         weight_decay=self.hparams['weight_decay'])

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs * 2, num_classes,
                                              self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                                          lr=self.hparams["lr"],
                                          weight_decay=self.hparams['weight_decay'])

        self.register_buffer('embeddings', torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(self.network_f.n_outputs, num_classes,
                                             self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(self.network_f.n_outputs, num_classes,
                                             self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"], weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(), 'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p**2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0


class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """
    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=False):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(), create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(), retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size == 2048 else input_feat_size * 2

        self.cdpl = nn.Sequential(nn.Linear(input_feat_size, hidden_size), nn.BatchNorm1d(hidden_size),
                                  nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                                  nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_size, input_feat_size), nn.BatchNorm1d(input_feat_size))

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex == val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end - ex) + ex
            shuffle_indices2 = torch.randperm(end - ex) + ex
            for idx in range(end - ex):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale * \
            (lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.hparams["lr"],
                                          weight_decay=self.hparams['weight_decay'],
                                          betas=betas)

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, epoch=None, delta_dir=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))
