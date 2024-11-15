# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:    https://github.com/facebookresearch/mae
# SatViT: https://github.com/antofuller/SatViT
# --------------------------------------------------------

# 加载BigEarthNet数据集（12通道数据），基于自监督训练 SatVit编解码器

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

from utils import misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from models.model_satvit import SatViT

from datasets.bigearthnet_dataset_seco_B12 import Bigearthnet
from datasets.bigearthnet_dataset_seco_lmdb_B12 import LMDBDataset


def get_args_parser():
    parser = argparse.ArgumentParser('SatVit pre-training', add_help=False)
    # Train parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_freq', default=20, type=int, help='print log freq')

    # Dataset parameters
    parser.add_argument('--lmdb', default=True, type=bool, help='use lmdb dataset')
    parser.add_argument('--lmdb_dir', default='/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S2/dataload_op1_lmdb', type=str, help='')
    parser.add_argument('--data_dir', default='/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S2', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output/pretrain_satvit_gate_v1', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/pretrain_satvit_gate_v1', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # Model parameters
    parser.add_argument('--model_mode', default='SatVit-V1', type=str, metavar='MODEL', help='SatVit-V1 or SatVit-V2')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--patch_hw', default=16, type=int, help='pixels per patch (both height and width)')
    parser.add_argument('--num_channels', default=12, type=int, help='total input bands')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
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


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, log_writer, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for iter, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if iter % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, iter / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)  # (bs, 12, 224, 224)

        batch_ins = rearrange(samples, 'b c (h i) (w j) -> b (h w) (c i j)', i=16, j=16)  # (bs, 196, 3072)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(batch_ins, mask_ratio=args.mask_ratio)

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
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # simple augmentation
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    
    train_transforms = cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor()
    ])

    if args.lmdb:
        dataset_train = LMDBDataset(
            lmdb_file=os.path.join(args.lmdb_dir, 'train_B12.lmdb'),
            transform=train_transforms)
    else:
        dataset_train = Bigearthnet(
            root=args.data_dir,
            split='train',
            bands=bands,
            use_new_labels=True,
            transform=train_transforms)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    patch_hw = args.patch_hw  # pixels per patch (both height and width)
    num_channels = args.num_channels  # total input bands
    io_dim = int(patch_hw * patch_hw * num_channels)
    patches_dim = int((args.input_size / patch_hw)**2)
    if args.model_mode == 'SatVit-V1':
        model = SatViT(in_dim=io_dim, out_dim=io_dim, num_patches=patches_dim, encoder_dim=768, encoder_depth=12, encoder_num_heads=12, decoder_dim=384, decoder_depth=2, decoder_num_heads=6
        )
    elif args.model_mode == 'SatVit-V2':
        model = SatViT(in_dim=io_dim, out_dim=io_dim, num_patches=patches_dim, encoder_dim=768, encoder_depth=12, encoder_num_heads=12, decoder_dim=512, decoder_depth=1, decoder_num_heads=8
        )
    summary(model, input_size=[(16, 196, 3072)], device="cuda")
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
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
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch=epoch)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)