# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from domainbed.networks import ResNet
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed, hp_dict=None):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST',
                    'ColoredMNIST', 'ColoredMNISTBack']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', hp_dict['resnet18'], lambda r: True)
    _hparam('resnet_dropout', 0.1, lambda r: r.choice([hp_dict['dpot']]))#
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([hp_dict['dpot']]))

    elif algorithm == 'Fish':  # or 'Contrast':
        _hparam('meta_lr', 0.5, lambda r: r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm in ['AT', 'UAT', 'nUAT','SUAT','KUAT','KAT2']:#!
        _hparam('at_eps', 1,
                lambda r: r.uniform(hp_dict['at_eps_lb'],hp_dict['at_eps_ub']))  # (-1, 0)
        _hparam('at_alpha', 0.1,
                lambda r: 10**r.uniform(hp_dict['at_alpha_lb'], hp_dict['at_alpha_ub']))
        _hparam('at_name', 0, lambda r: r.randint(10000))
        
    elif algorithm in ['CBAT','LDAT','LAT']:#!
        _hparam('at_eps', 1,
                lambda r: r.uniform(hp_dict['at_eps_lb'],hp_dict['at_eps_ub']))  # (-1, 0)
        _hparam('at_alpha', 0.1,
                lambda r: 10**r.uniform(hp_dict['at_alpha_lb'], hp_dict['at_alpha_ub']))
        _hparam('at_name', 0, lambda r: r.randint(10000))

    elif algorithm in ['MAT','LDMAT']:#!
        _hparam('at_eps', 1,
                lambda r: r.uniform(hp_dict['at_eps_lb'],hp_dict['at_eps_ub']))  # (-1, 0)
        _hparam('at_alpha', 0.1,
                lambda r: 10**r.uniform(hp_dict['at_alpha_lb'], hp_dict['at_alpha_ub']))
        _hparam('at_name', 0, lambda r: r.randint(10000))

    elif algorithm in ['LAT']:#!
        _hparam('at_eps', 1,
                lambda r: r.uniform(hp_dict['at_eps_lb'],hp_dict['at_eps_ub']))  # (-1, 0)
        _hparam('at_alpha', 0.1,
                lambda r: 10**r.uniform(hp_dict['at_alpha_lb'], hp_dict['at_alpha_ub']))
        _hparam('at_name', 0, lambda r: r.randint(10000))
    
    elif algorithm in ['CKAT']:#!
        _hparam('at_eps', 1,
                lambda r: r.uniform(hp_dict['at_eps_lb'],hp_dict['at_eps_ub']))  # (-1, 0)
        _hparam('at_alpha', 0.1,
                lambda r: 10**r.uniform(hp_dict['at_alpha_lb'], hp_dict['at_alpha_ub']))
        _hparam('at_name', 0, lambda r: r.randint(10000))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm in ["VREx", "VRExAT"]:
        _hparam('vrex_lambda', 1e3, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: int(10**r.uniform(-3, 5)))

    if algorithm in ["KUAT","MAT","CKAT",'KAT2','LDMAT']:
        _hparam('at_delta_num', 8 , lambda r: r.uniform(hp_dict['kdelta_lower_bound'],hp_dict['kdelta_upper_bound']))#!delta����
        _hparam('kat_alpha_step', 1,lambda r: 10**r.uniform(hp_dict['kat_alpha_step_lb'], hp_dict['kat_alpha_step_ub']))
    if algorithm in ["CBAT","LDAT",'LDMAT','LAT']:
        _hparam('at_cb_rank', 8, lambda r: r.uniform(hp_dict['cb_rank_lb'],hp_dict['cb_rank_ub']))#!codebook rank �Ͻ�
        _hparam('A_lr',0.1,lambda r: 10**r.uniform(hp_dict['A_lr_lb'],hp_dict['A_lr_ub']))
        _hparam('B_lr',0.1,lambda r: 10**r.uniform(hp_dict['B_lr_lb'],hp_dict['B_lr_ub']))
    if algorithm in ['LAT']:
        _hparam('local_loss_rate',0.1,lambda r: r.uniform(hp_dict['llr_lb'],hp_dict['llr_ub']))
        _hparam('entropy_maximization_loss_rate',0.1,lambda r: r.uniform(hp_dict['emlr_lb'],hp_dict['emlr_ub']))
    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    elif algorithm in ['DCAT']:#!
        _hparam('at_eps', 1,
                lambda r: 10**r.uniform(-2, 0))  # (-1, 0)
        _hparam('at_alpha', 0.1,#defualt 0.1
                lambda r: 10**r.uniform(-2, 1))
        _hparam('at_name', 0, lambda r: r.randint(10000))#?
        
    elif algorithm in ['DCAT2']:#!
        _hparam('at_eps', 1,
                lambda r: 10**r.uniform(0, 2))  # (-1, 0)
        _hparam('at_alpha', 0.1,#defualt 0.1
                lambda r: 10**r.uniform(-2, 1))
        _hparam('at_name', 0, lambda r: r.randint(10000))#?

    if dataset in SMALL_IMAGES:
        _hparam('is_cmnist', 1, lambda r: 1)
    else:
        _hparam('is_cmnist', 0, lambda r: 0)

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: r.uniform(hp_dict['lr_lb'], hp_dict['lr_ub']))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: r.uniform(hp_dict['wd_lb'], hp_dict['wd_ub']))
    else:
        _hparam('weight_decay', 0., lambda r: r.uniform(hp_dict['wd_lb'], hp_dict['wd_ub']))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(r.uniform(hp_dict['bsz_lb'], hp_dict['bsz_ub'])))

    if dataset in SMALL_IMAGES:#!
        _hparam('at_rank',10, lambda r: int(r.uniform(8,12)))
    else:
        _hparam('at_rank',50, lambda r: int(r.uniform(30,70)))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES: 
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    if algorithm in ['Alg3', 'Alg5']:
        _hparam('lam', 0.1405773046125034, lambda r: 10 **
                r.uniform(-1, 0))  # 0.2 2 (-1.5, -0.3)

    # if algorithm in ['Alg3', 'Alg4']:
    #     _hparam('k', 0.01, lambda r: 10**r.uniform(0, 2)) # 3 1

    return hparams


def default_hparams(algorithm, dataset, hp_dict):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0, hp_dict).items()}


def random_hparams(algorithm, dataset, seed, hp_dict):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed, hp_dict).items()}
