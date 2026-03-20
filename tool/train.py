"""
train.py
Custom extension of Point Transformer (POSTECH-CVLab)
Original implementation: https://github.com/POSTECH-CVLab/point-transformer

Author: Alexandre Devaux Rivière
Project: NPM3D
Date: 20/03/2026
"""

import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.modelnet40 import ModelNet40
from util.common_util import AverageMeter, find_free_port
from util.data_util import collate_fn
from util import transform as t

import csv
from util.profiler import latency_profiler

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification')
    parser.add_argument('--config', type=str, default='config/modelnet40/modelnet40_pointtransformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/modelnet40/modelnet40_pointtransformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 'modelnet40':
        ModelNet40(split='train', data_root=args.data_root)
        ModelNet40(split='test', data_root=args.data_root)
    else:
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_acc
    args, best_acc = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.arch == 'pointtransformer_cls':
        from model.pointtransformer.pointtransformer_cls import pointtransformer_cls as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(
        c=args.fea_dim, k=args.classes, num_neighbors_k=args.num_neighbors_k,
        pos_enc=args.pos_enc, attn_type=args.attn_type
    )
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # TODO
    #train_transform = t.Compose([
    #    t.RandomScale([0.85, 1.15]),
    #    t.RandomRotate(),
    #    t.RandomJitter(sigma=0.01)
    #])

    train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    train_data = ModelNet40(split='train', data_root=args.data_root, transform=train_transform, loop=args.loop)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss().cuda()

    opt_name = args.optimizer_name
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif opt_name == 'madgrad':
        import madgrad
        optimizer = madgrad.MADGRAD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer not supported: {opt_name}")

    sched_name = args.scheduler_name
    if sched_name == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
    elif sched_name == 'onecycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.base_lr, steps_per_epoch=len(train_loader), epochs=args.epochs, final_div_factor=100.0)
    elif sched_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        raise ValueError(f"Scheduler not supported: {sched_name}")

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("Optimizer: {}".format(args.optimizer_name))
        logger.info("Scheduler: {}".format(args.scheduler_name))
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            #best_acc = 40.0
            best_acc = checkpoint['best_acc']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))

    val_loader = None
    if args.evaluate:
        val_transform = None
        val_data = ModelNet40(split='test', data_root=args.data_root, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mAcc_train, OA_train = train(train_loader, model, criterion, optimizer, epoch, scheduler)
        current_lr = optimizer.param_groups[0]['lr']

        if sched_name != 'onecycle':
            scheduler.step()

        epoch_log = epoch + 1
        if main_process():
            logger.info(f"Epoch: [{epoch_log}/{args.epochs}] Current LR: {current_lr:.6f}")
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('OA_train', OA_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, mAcc_val, OA_val = validate(val_loader, model, criterion, epoch_log)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', OA_val, epoch_log)
                is_best = OA_val > best_acc
                best_acc = max(best_acc, OA_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_acc': best_acc, 'is_best': is_best}, filename)
            if is_best:
                logger.info('Best validation mAcc updated to: {:.4f}'.format(best_acc))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Acc: %.3f' % (best_acc))


def train(train_loader, model, criterion, optimizer, epoch, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    class_correct = np.zeros(args.classes)
    class_total = np.zeros(args.classes)
    total_correct = 0
    total_samples = 0

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    sched_name = args.scheduler_name
    optim_name = args.optimizer_name

    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        output = model([coord, feat, offset])
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()

        if scheduler is not None and sched_name == 'onecycle' and optim_name == 'adamw':
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            current_iter = epoch * len(train_loader) + i + 1
            if current_iter % 10 == 0 and main_process():
                writer.add_scalar('grad_norm_before_clip', total_norm, current_iter)

        optimizer.step()

        pred = output.max(1)[1]
        correct = pred.eq(target).cpu()

        if scheduler is not None and sched_name == 'onecycle':
            scheduler.step()

        for c in range(args.classes):
            class_mask = (target.cpu() == c)
            class_correct[c] += correct[class_mask].sum().item()
            class_total[c] += class_mask.sum().item()

        total_correct += correct.sum().item()
        total_samples += target.size(0)

        loss_meter.update(loss.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            accuracy = total_correct / total_samples
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            accuracy = total_correct / total_samples
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('acc_train_batch', accuracy, current_iter)

    OA = total_correct / total_samples
    class_accs = class_correct / (class_total + 1e-10)
    mAcc = np.mean(class_accs)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mAcc/OA {:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mAcc, OA))

    return loss_meter.avg, mAcc, OA


def validate(val_loader, model, criterion, epoch):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    latency_profiler.enabled = args.profile
    if args.profile:
        latency_profiler.reset()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    class_correct = np.zeros(args.classes)
    class_total = np.zeros(args.classes)
    total_correct = 0
    total_samples = 0

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        latency_profiler.on_batch_start()
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        with torch.no_grad():
            output = model([coord, feat, offset])
        loss = criterion(output, target)

        pred = output.max(1)[1]
        correct = pred.eq(target).cpu()

        for c in range(args.classes):
            class_mask = (target.cpu() == c)
            class_correct[c] += correct[class_mask].sum().item()
            class_total[c] += class_mask.sum().item()

        total_correct += correct.sum().item()
        total_samples += target.size(0)

        loss_meter.update(loss.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 and main_process():
            accuracy = total_correct / total_samples
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    OA = total_correct / total_samples
    class_accs = class_correct / (class_total + 1e-10)
    mAcc = np.mean(class_accs)
    latency_profiler.enabled = False

    if main_process():
        if args.profile:
            csv_path = os.path.join(args.save_path, 'inference_latency_ms.csv')
            latency_profiler.save_csv(csv_path, epoch)
            latency_profiler.log_summary(logger)

        logger.info('Val result: mAcc/OA {:.4f}/{:.4f}.'.format(mAcc, OA))
        for i in range(args.classes):
            logger.info('Class_{} Result: accuracy {:.4f}.'.format(i, class_accs[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mAcc, OA


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
