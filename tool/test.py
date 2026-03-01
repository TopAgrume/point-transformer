import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
# updated
from util.modelnet40 import ModelNet40
from util.data_util import collate_fn

random.seed(123)
np.random.seed(123)

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


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro': # updated
        from model.pointtransformer.pointtransformer_cls import pointtransformer_cls as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda() # updated

    # updated
    if args.names_path and os.path.exists(args.names_path):
        names = [line.rstrip('\n') for line in open(args.names_path)]
    else:
        names = [str(i) for i in range(args.classes)]

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)

def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def test(model, criterion, names): # updated
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model.eval()
    check_makedirs(args.save_folder)

    test_data = ModelNet40(split='test', data_root=args.data_root, transform=None)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True, collate_fn=collate_fn)

    class_correct = np.zeros(args.classes)
    class_total = np.zeros(args.classes)
    total_correct = 0
    total_samples = 0

    pred_save, label_save = [], []
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(test_loader):
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)

        with torch.no_grad():
            output = model([coord, feat, offset])

        pred = output.max(1)[1]
        correct = pred.eq(target).cpu()

        pred_save.append(pred.cpu().numpy())
        label_save.append(target.cpu().numpy())

        for c in range(args.classes):
            class_mask = (target.cpu() == c)
            class_correct[c] += correct[class_mask].sum().item()
            class_total[c] += class_mask.sum().item()

        total_correct += correct.sum().item()
        total_samples += target.size(0)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
            accuracy = total_correct / total_samples
            logger.info('Test: [{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader), batch_time=batch_time, accuracy=accuracy))

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': np.concatenate(pred_save)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': np.concatenate(label_save)}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    OA = total_correct / total_samples
    class_accs = class_correct / (class_total + 1e-10)
    mAcc = np.mean(class_accs)

    logger.info('Val result: mAcc/OA {:.4f}/{:.4f}.'.format(mAcc, OA))
    for i in range(args.classes):
        logger.info('Class_{} Result: accuracy {:.4f}, name: {}.'.format(i, class_accs[i], names[i] if i < len(names) else "Unknown"))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

if __name__ == '__main__':
    main()
