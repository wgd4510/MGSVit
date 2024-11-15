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
from sklearn.metrics import average_precision_score
from utils.log import load_logger
from utils import misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.lr_sched import adjust_learning_rate
from datasets.bigearthnet_dataset_seco_lmdb_B14 import LMDBDataset_S1_S2, LMDBDataset_S1, LMDBDataset_S2
from ours_model_linear import MCMAE_Encoder, MCMAE_Encoder_single


def get_args_parser():
    parser = argparse.ArgumentParser('MoCoMae linear classification training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--print_freq', default=20, type=int, help='print log freq')
    parser.add_argument('--accum_iter', default=1, type=int, 
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--output_dir', default='./output/linear_test', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--lmdb_name', default='B14', type=str, help='S1:B2; S2:B12; S1+S2:B14')

    # Finetuning and inference parameters
    parser.add_argument('--s1_weights', default=None, help='S1 finetune checkpoint')
    parser.add_argument('--s2_weights', default=None, help='S2 finetune checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument("--freeze", action='store_true', help="freeze encoder module or not")
    parser.add_argument('--num_workers', default=10, type=int)

    # Model parameters
    parser.add_argument('--model_mode', default='MoCoMae-V1', type=str, metavar='MODEL', help='SatVit-V1 or SatVit-V2')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--patch_hw', default=16, type=int, help='pixels per patch (both height and width)')
    parser.add_argument('--shortnum', default=4, type=int, help='short num')
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
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
    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor()])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(256),
            cvtransforms.CenterCrop(224),
            cvtransforms.ToTensor(),
            ])
    if args.lmdb_name == 'B2':
        lmdb_train = '/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S1/dataload_op1_lmdb/train_B2.lmdb'
        lmdb_val = '/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S1/dataload_op1_lmdb/val_B2.lmdb'
        bands = ['VH', 'VV']
        n_channels = 2
        dataset_train = LMDBDataset_S1(lmdb_file=lmdb_train, transform=train_transforms)  # 311667
        dataset_val = LMDBDataset_S1(lmdb_file=lmdb_val, transform=val_transforms)  # 103944
    elif args.lmdb_name == 'B12':
        lmdb_train = '/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S2/dataload_op1_lmdb/train_B12.lmdb'
        lmdb_val = '/home/wgd/code/Datasets/BigEarthNet/BigEarthNet-S2/dataload_op1_lmdb/val_B12.lmdb'
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        n_channels = 12
        dataset_train = LMDBDataset_S2(lmdb_file=lmdb_train, transform=train_transforms)  # 311667
        dataset_val = LMDBDataset_S2(lmdb_file=lmdb_val, transform=val_transforms)  # 103944
    else:
        lmdb_train = '/home/wgd/code/Datasets/BigEarthNet/lmdb_data/train_B12_B2.lmdb'
        lmdb_val = '/home/wgd/code/Datasets/BigEarthNet/lmdb_data/val_B12_B2.lmdb'
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'VH', 'VV']
        n_channels = 14
        dataset_train = LMDBDataset_S1_S2(lmdb_file=lmdb_train, transform=train_transforms)  # 311667
        dataset_val = LMDBDataset_S1_S2(lmdb_file=lmdb_val, transform=val_transforms)  # 103944

    return dataset_train, dataset_val, n_channels


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, max_norm=0, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    for iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if iter % accum_iter == 0:
            adjust_learning_rate(optimizer, iter / len(data_loader) + epoch, args)

        targets = targets.to(device, non_blocking=True)
        if args.lmdb_name == 'B2' or args.lmdb_name == 'B12':
            samples = samples.to(device, non_blocking=True)
            batch_ins = rearrange(samples, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)
        else:
            samples_s2 = samples[:, :12, :, :]
            samples_s1 = samples[:, 12:, :, :]
            samples_s1 = samples_s1.to(device, non_blocking=True)
            samples_s2 = samples_s2.to(device, non_blocking=True)
            batch_ins_s1 = rearrange(samples_s1, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)
            batch_ins_s2 = rearrange(samples_s2, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)

        with torch.cuda.amp.autocast():
            if args.lmdb_name == 'B2' or args.lmdb_name == 'B12':
                outputs = model(batch_ins)
            else:
                outputs = model(batch_ins_s1, batch_ins_s2)

            loss = criterion(outputs, targets.long())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(iter + 1) % accum_iter == 0)
        if (iter + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (iter + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((iter / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, args=None, top5=False):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    # for iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for samples, target in metric_logger.log_every(data_loader, 10, header):
        if args.lmdb_name == 'B2' or args.lmdb_name == 'B12':
            samples = samples.to(device, non_blocking=True)
            batch_ins = rearrange(samples, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)
        else:
            samples_s2 = samples[:, :12, :, :]
            samples_s1 = samples[:, 12:, :, :]
            samples_s1 = samples_s1.to(device, non_blocking=True)
            samples_s2 = samples_s2.to(device, non_blocking=True)
        
            batch_ins_s1 = rearrange(samples_s1, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)
            batch_ins_s2 = rearrange(samples_s2, 'b c (h i) (w j) -> b (h w) (c i j)', i=args.patch_hw, j=args.patch_hw)  # (bs, 196, 3072)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            if args.lmdb_name == 'B2' or args.lmdb_name == 'B12':
                outputs = model(batch_ins)
            else:
                outputs = model(batch_ins_s1, batch_ins_s2)
            loss = criterion(outputs, target.long())

        score = torch.sigmoid(outputs).detach().cpu()
        acc1 = average_precision_score(target.cpu(), score, average='micro') * 100.0
        if top5:
            acc5 = acc1

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if top5: 
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if top5: 
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, losses=metric_logger.loss))
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

    dataset_train, dataset_val, n_channels = build_lmdb_dataset(args)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    patch_hw = args.patch_hw  # pixels per patch (both height and width)
    io_dim_s1 = int(patch_hw * patch_hw * 2)
    io_dim_s2 = int(patch_hw * patch_hw * 12)
    patches_dim = int((args.input_size / patch_hw)**2)
    shortnum = args.shortnum
    if n_channels == 2:
        model = MCMAE_Encoder_single(
            io_dim=io_dim_s1, 
            weights=args.s1_weights,
            num_patches=patches_dim,
            encoder_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            shortnum=shortnum,
            num_classes=19)
    elif n_channels == 12:
        model = MCMAE_Encoder_single(
            io_dim=io_dim_s2, 
            weights=args.s2_weights,
            num_patches=patches_dim,
            encoder_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            shortnum=shortnum,
            num_classes=19)
    else:
        model = MCMAE_Encoder(
            io_dim_s1=io_dim_s1, 
            io_dim_s2=io_dim_s2,
            S1_weights=args.s1_weights,
            S2_weights=args.s2_weights,
            num_patches=patches_dim,
            encoder_dim=768,
            encoder_depth=24,
            encoder_num_heads=12,
            shortnum=shortnum,
            num_classes=19
            )

    # 选择是否要冻结不需要训练的层
    if args.freeze:
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.gmu.named_parameters():
            p.requires_grad = True
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    # 如果resume不为空，则继续训练中断的模型, 并且先评估一下模型，不然后面会报test_stats未初始化
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # 如果评估模型，则评估完直接退出程序，不进行后续的训练
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, criterion, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    
    if misc.is_main_process():
        log.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        log.info("actual lr: %.2e" % args.lr)
        log.info("accumulate grad iterations: %d" % args.accum_iter)
        log.info("effective batch size: %d" % eff_batch_size)
        log.info("world_size = %d" % misc.get_world_size())
        log.info("number of params (M): %.2f" % (n_parameters / 1.e6))
        log.info("criterion = %s" % str(criterion))
        log.info(optimizer)
        log.info(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )

        if epoch % 5 == 0 or (epoch + 1 == args.epochs):
            if args.output_dir:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

            test_stats = evaluate(data_loader_val, model, device, criterion, args)
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            # log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {'epoch': epoch, 
                     **{f'train_{k}': f'{v:.6f}' for k, v in train_stats.items()},
                     **{f'test_{k}': f'{v:.6f}' for k, v in test_stats.items()},
                     'Max accuracy': '{:.6f}'.format(max_accuracy),
                     'n_parameters': n_parameters}
        
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
