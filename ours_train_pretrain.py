# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MoCo_v3 https://github.com/facebookresearch/moco-v3
# MAE:    https://github.com/facebookresearch/mae
# SatViT: https://github.com/antofuller/SatViT
# MixMAE: https://github.com/Sense-X/MixMIM
# --------------------------------------------------------

import argparse
import numpy as np
import os
import datetime
import time
from pathlib import Path
import math
import sys
import json
from einops import rearrange
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from cvtorchvision import cvtransforms
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from torchinfo import summary

from utils.log import load_logger
from utils import misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.lr_sched import adjust_learning_rate, adjust_moco_momentum

from ours_model_pretrain import base_encoder, MCMAE

def get_args_parser():
    parser = argparse.ArgumentParser('MoCoMae pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--print_freq', default=20, type=int, help='print log freq')
    parser.add_argument('--accum_iter', default=1, type=int, 
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--output_dir', default='./output/test', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Dataset parameters
    parser.add_argument('--lmdb_name', default='B14', type=str, help='S1:B2; S2:B12; S1+S2:B14')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dtype', default='float', type=str, help='data :uint8 or float32')
    
    # Model parameters
    parser.add_argument('--model_mode', default='MoCoMae-V1', type=str, metavar='MODEL', help='SatVit-V1 or SatVit-V2')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--patch_hw', default=16, type=int, help='pixels per patch (both height and width)')
    parser.add_argument('--shortnum', default=4, type=int, help='short num')
    parser.add_argument('--num_channels', default=12, type=int, help='total input bands')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a half-cycle cosine schedule')
    parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()
    return args


def build_lmdb_dataset(args):

    class TwoCropsTransform:
        """Take two random crops of one image"""
        def __init__(self, base_transform1, base_transform2):
            self.base_transform1 = base_transform1
            self.base_transform2 = base_transform2

        def __call__(self, x):
            im1 = self.base_transform1(x)
            im2 = self.base_transform2(x)
            return [im1, im2]
        
    if args.dtype=='uint8':
        from datasets.rs_transforms_uint8 import RandomBrightness, RandomContrast, ToGray, GaussianBlur
    else:
        from datasets.rs_transforms_float32 import RandomBrightness, RandomContrast, ToGray, GaussianBlur
 
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

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        cvtransforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.)),
        cvtransforms.RandomApply([RandomBrightness(0.4), RandomContrast(0.4)], p=0.8),
        cvtransforms.RandomApply([ToGray(n_channels)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor()
    ]

    augmentation2 = [
        cvtransforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.)),
        cvtransforms.RandomApply([RandomBrightness(0.4), RandomContrast(0.4)], p=0.8),
        cvtransforms.RandomApply([ToGray(n_channels)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        cvtransforms.RandomHorizontalFlip(),  
        cvtransforms.ToTensor()
    ]
    train_transforms1 = cvtransforms.Compose(augmentation1)
    train_transforms2 = cvtransforms.Compose(augmentation2)

    train_dataset = LMDBDataset(lmdb_file=lmdb_train, transform=TwoCropsTransform(train_transforms1, train_transforms2))
    return train_dataset, n_channels


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, log_writer, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    moco_m = args.moco_m
    for iter, (images, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if iter % accum_iter == 0:
            adjust_learning_rate(optimizer, iter / len(data_loader) + epoch, args)
            if args.moco_m_cos:
                moco_m = adjust_moco_momentum(iter / len(data_loader) + epoch, args)

        samples_x1 = images[0]
        samples_x2 = images[1]
        samples_x1 = samples_x1.type(torch.float).to(device, non_blocking=True)
        samples_x2 = samples_x2.type(torch.float).to(device, non_blocking=True)

        batch_x1 = rearrange(samples_x1, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)
        batch_x2 = rearrange(samples_x2, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(batch_x1, batch_x2, moco_m, args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(iter + 1) % accum_iter == 0)
        if (iter + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (iter + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((iter / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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

    patch_hw = args.patch_hw  # pixels per patch (both height and width)
    io_dim = int(patch_hw * patch_hw * num_channels)
    patches_dim = int((args.input_size / patch_hw)**2)
    shortnum = args.shortnum
    model = MCMAE(
        in_dim=io_dim, 
        out_dim=io_dim, 
        base_encoder=base_encoder, 
        num_patches=patches_dim, 
        encoder_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12, 
        decoder_dim=384, 
        decoder_depth=2, 
        decoder_num_heads=6, 
        shortnum=shortnum, 
        moco_dim=256, 
        moco_mlp_dim=4096, 
        T=args.moco_t)
    
    # summary(model, input_size=[(8, patches_dim, io_dim), (8, patches_dim, io_dim)], device="cuda", m=args.moco_m, mask_ratio=0.75) 

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # 如果resume不为空，则继续训练中断的模型, 并且先评估一下模型，不然后面会报test_stats未初始化
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