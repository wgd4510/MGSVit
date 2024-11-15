# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import models.mae.util.misc as misc
from models.mae.util.misc import NativeScalerWithGradNormCount as NativeScaler


import models.mae.models_mae as models_mae
from models.mae.engine_pretrain import train_one_epoch

from utils.log import load_logger
from cvtorchvision import cvtransforms


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir/test', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # new   
    parser.add_argument('--lmdb_name', default='B2', type=str, help='S1:B2; S2:B12; S1+S2:B14')

    args = parser.parse_args()
    return args


def build_lmdb_dataset(args):

    if args.lmdb_name == 'B2':
        from datasets.bigearthnet_dataset_seco_lmdb_B14 import LMDBDataset_S1 as LMDBDataset
        bands = ['VH', 'VV']
        n_channels = 2
        lmdb_train = '/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S1/dataload_op1_lmdb/train_B2.lmdb'
    elif args.lmdb_name == 'B12':
        from datasets.bigearthnet_dataset_seco_lmdb_B14 import LMDBDataset_S2 as LMDBDataset
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        n_channels = 12
        lmdb_train = '/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S2/dataload_op1_lmdb/train_B12.lmdb'
    else:
        from datasets.bigearthnet_dataset_seco_lmdb_B14 import LMDBDataset_S1_S2 as LMDBDataset
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'VH', 'VV']
        n_channels = 14
        lmdb_train = '/home/wgd/code/Datasets/BigEarthNet/lmdb_data/train_B12_B2.lmdb'

    transform_train = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),  # 3 is bicubic
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor(),
            ])
    
    train_dataset = LMDBDataset(lmdb_file=lmdb_train, transform=transform_train)
    return train_dataset, n_channels
    

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log = load_logger(args)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None
        
    dataset_train, num_channels = build_lmdb_dataset(args)

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, in_chans=num_channels)
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if misc.is_main_process():
        log.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        log.info("actual lr: %.2e" % args.lr)
        log.info("accumulate grad iterations: %d" % args.accum_iter)
        log.info("effective batch size: %d" % eff_batch_size)
        log.info(optimizer)
        log.info(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log.info(json.dumps(log_stats))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        log.info('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    args = get_args_parser()
    main(args)