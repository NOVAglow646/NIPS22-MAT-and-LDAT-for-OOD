# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex


class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['CUDA_VISIBLE_DEVICES=0',
                   'python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
                    self.train_args['algorithm'],
                    self.train_args['test_envs'],
                    self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def all_test_env_combinationsNew(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    yield [2]


def all_test_env_combinationsNICO(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    yield [2, 3]


def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]


def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
                   data_dir, task, holdout_fraction, single_test_envs, hparams,gpus,is_cmnist,
                   kdelta_lower_bound,kdelta_upper_bound,
                   at_eps_lb,at_eps_ub,at_alpha_lb,at_alpha_ub,
                   cb_rank_lb,cb_rank_ub,
                   kat_alpha_step_lb,kat_alpha_step_ub,
                   B_lr_lb,B_lr_ub,
                   A_lr_lb,A_lr_ub,
                   KAT_num_iter,CBAT_num_iter,CBAT2_num_iter,
                   lr_lb,lr_ub,bsz_lb,bsz_ub,
                   wd_lb,wd_ub,dpot,
                   res18, test_domains, visualization_dir,
                   llr_lb,llr_ub,emlr_lb,emlr_ub):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                #print(datasets.num_environments(dataset))
                if single_test_envs:
                    all_test_envs = [[i] for i in range(datasets.num_environments(dataset))]#
                    ##[2]
                else:
                    all_test_envs=[]
                    for i in range(0,len(test_domains),2):
                        all_test_envs.append(int(test_domains[i]))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                                                            algorithm, test_envs, hparams_seed, trial_seed)
                        train_args['gpus']=gpus
                        train_args['is_cmnist']=is_cmnist
                        train_args['kdelta_lower_bound']=kdelta_lower_bound
                        train_args['kdelta_upper_bound']=kdelta_upper_bound
                        train_args['at_eps_lb']=at_eps_lb
                        train_args['at_eps_ub']=at_eps_ub
                        train_args['at_alpha_lb']=at_alpha_lb
                        train_args['at_alpha_ub']=at_alpha_ub
                        train_args['cb_rank_lb']=cb_rank_lb
                        train_args['cb_rank_ub']=cb_rank_ub
                        train_args['kat_alpha_step_lb']=kat_alpha_step_lb
                        train_args['kat_alpha_step_ub']=kat_alpha_step_ub
                        train_args['B_lr_lb']=B_lr_lb
                        train_args['B_lr_ub']=B_lr_ub
                        train_args['A_lr_lb']=A_lr_lb
                        train_args['A_lr_ub']=A_lr_ub
                        train_args['KAT_num_iter']=KAT_num_iter
                        train_args['CBAT_num_iter']=CBAT_num_iter
                        train_args['CBAT2_num_iter']=CBAT2_num_iter
                        train_args['lr_lb']=lr_lb
                        train_args['lr_ub']=lr_ub
                        train_args['bsz_lb']=bsz_lb
                        train_args['bsz_ub']=bsz_ub                        
                        train_args['wd_lb']=wd_lb
                        train_args['wd_ub']=wd_ub
                        train_args['dpot']=dpot
                        train_args['resnet18']=res18
                        train_args['visualization_dir']=visualization_dir
                        train_args['llr_lb']=llr_lb
                        train_args['llr_ub']=llr_ub
                        train_args['emlr_lb']=emlr_lb
                        train_args['emlr_ub']=emlr_ub
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list


def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str,
                        default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--gpus',default='0',type=str)
    parser.add_argument('--is_cmnist',default=0,type=int)
    parser.add_argument('--kdelta_lower_bound',default=4,type=int)
    parser.add_argument('--kdelta_upper_bound',default=8,type=int)
    parser.add_argument('--at_eps_lb',default=0.1,type=float)
    parser.add_argument('--at_eps_ub',default=0.1,type=float)
    parser.add_argument('--at_alpha_lb',default=-2,type=int)
    parser.add_argument('--at_alpha_ub',default=1,type=int)
    parser.add_argument('--cb_rank_lb',default=4,type=int)
    parser.add_argument('--cb_rank_ub',default=8,type=int)
    parser.add_argument('--kat_alpha_step_lb',default=-2,type=int)
    parser.add_argument('--kat_alpha_step_ub',default=1,type=int)
    parser.add_argument('--B_lr_lb',default=-2,type=int)
    parser.add_argument('--B_lr_ub',default=1,type=int)
    parser.add_argument('--A_lr_lb',default=-2,type=int)
    parser.add_argument('--A_lr_ub',default=1,type=int)
    parser.add_argument('--KAT_num_iter',default=1,type=int)
    parser.add_argument('--CBAT_num_iter',default=1,type=int)
    parser.add_argument('--CBAT2_num_iter',default=1,type=int)
    parser.add_argument('--lr_lb',default=5e-5,type=float)
    parser.add_argument('--lr_ub',default=5e-5,type=float)
    parser.add_argument('--bsz_lb',default=32,type=int)
    parser.add_argument('--bsz_ub',default=32,type=int)
    parser.add_argument('--wd_lb',default=1e-6,type=float)
    parser.add_argument('--wd_ub',default=1e-6,type=float)
    parser.add_argument('--dpot',default=0.1,type=float)
    parser.add_argument('--resnet18',default=1,type=int)
    parser.add_argument('--test_domains',default='0',type=str) 
    parser.add_argument('--visualization_dir',default='',type=str)
    parser.add_argument('--llr_lb',default=0.1,type=float)
    parser.add_argument('--llr_ub',default=0.1,type=float)
    parser.add_argument('--emlr_lb',default=0.1,type=float)
    parser.add_argument('--emlr_ub',default=0.1,type=float)

    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams,
        gpus=args.gpus,
        is_cmnist=args.is_cmnist,
        kdelta_lower_bound=args.kdelta_lower_bound,
        kdelta_upper_bound=args.kdelta_upper_bound,
        at_eps_lb=args.at_eps_lb,
        at_eps_ub=args.at_eps_ub,
        at_alpha_lb=args.at_alpha_lb,
        at_alpha_ub=args.at_alpha_ub,
        cb_rank_lb=args.cb_rank_lb,
        cb_rank_ub=args.cb_rank_ub,
        kat_alpha_step_lb=args.kat_alpha_step_lb,
        kat_alpha_step_ub=args.kat_alpha_step_ub,
        B_lr_lb=args.B_lr_lb,
        B_lr_ub=args.B_lr_ub,
        A_lr_lb=args.A_lr_lb,
        A_lr_ub=args.A_lr_ub,
        KAT_num_iter=args.KAT_num_iter,
        CBAT_num_iter=args.CBAT_num_iter,
        CBAT2_num_iter=args.CBAT2_num_iter,
        lr_lb=args.lr_lb,
        lr_ub=args.lr_ub,
        bsz_lb=args.bsz_lb,
        bsz_ub=args.bsz_ub,
        wd_lb=args.wd_lb,
        wd_ub=args.wd_ub,
        dpot=args.dpot,
        res18=args.resnet18,
        test_domains=args.test_domains,
        visualization_dir=args.visualization_dir,
        llr_lb=args.llr_lb,
        llr_ub=args.llr_ub,
        emlr_lb=args.emlr_lb,
        emlr_ub=args.emlr_ub
    )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
