import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import transforms

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils
import random
from plot_curve import plot_images
from PIL import ImageDraw
import cv2


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None, tb_writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    # plot loss
    if tb_writer:
        tags = ['classification', 'bbox_regression']  # params
        for x, tag in zip([metric_logger.meters["classification"].global_avg,
                           metric_logger.meters["bbox_regression"].global_avg], tags):
            tb_writer.add_scalar(tag, x, epoch)  # tensorboard

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def train_datasets_eda(datasets, tb_writer, save_dir, device, batch=9):
    imgs = []
    targets = []
    data_transform = transforms.Compose([transforms.Resize(),
                                         transforms.ToTensor()])

    for _ in range(batch):
        idx = random.randint(0, datasets.__len__() - 1)
        img, target = data_transform(*datasets.pull_item(idx))
        imgs.append(img)
        targets.append(torch.cat([torch.tensor([[_]]*target["labels"].size(0)), target["labels"].unsqueeze(1), target["boxes"]], dim=1))

    imgs = torch.stack(imgs, dim=0)
    targets = torch.cat(targets, dim=0)

    # plot batch images
    mosaic = plot_images(imgs, targets, ["person"]*batch, save_dir / "images.jpg")
    tb_writer.add_image("label", mosaic, dataformats="HWC")

    # plot mosaic
    mosaic_image, mosaic_label = datasets[random.randint(0, datasets.__len__() - 1)]
    image_array = mosaic_image.numpy()  # 将tensor数据转为numpy数据
    image_array = image_array * 255 / image_array.max()  # normalize，将图像数据扩展到[0,255]
    mat = cv2.cvtColor(np.uint8(image_array).transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

    for box in mosaic_label["boxes"]:
        cv2.rectangle(mat, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255))
    tb_writer.add_image("mosaic", mat, dataformats="HWC")
