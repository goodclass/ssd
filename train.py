import datetime
import os
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ssd import SSD300
from nets.ssd_training import (MultiboxLoss, get_lr_scheduler,
                               set_optimizer_lr, weights_init)
from utils.anchors import get_anchors
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    Cuda = False

    classes_path = 'conf/voc_classes.txt'
    model_path = 'model_data/ssd_weights.pth'
    input_shape = [300, 300]

    backbone = "vgg"
    pretrained = False
    #  [21, 45, 99, 153, 207, 261, 315]
    anchors_size = [30, 60, 111, 162, 213, 264, 315]

    Init_Epoch = 0
    End_Epoch = 200
    batch_size = 1

    Init_lr = 2e-3
    Min_lr = Init_lr * 0.01
    momentum = 0.937
    weight_decay = 5e-4

    # 使用到的学习率下降方式，可选的有'step'、'cos'
    lr_decay_type = 'cos'
    save_period = 10
    save_dir = 'logs'

    eval_flag = True
    eval_period = 10

    num_workers = 0
    # 训练与验证图片路径和标签
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, anchors_size, backbone)

    model = SSD300(num_classes, backbone, pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))

        # 根据预训练权重的Key和模型的Key进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 获得损失函数
    criterion = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    # 记录Loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    model_train = model
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(classes_path=classes_path, model_path=model_path, input_shape=input_shape, Init_Epoch=Init_Epoch,
                End_Epoch=End_Epoch, batch_size=batch_size, Init_lr=Init_lr, Min_lr=Min_lr,
                momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir,
                num_workers=num_workers, num_train=num_train, num_val=num_val)

    if True:
        #  判断当前batch_size，自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3
        lr_limit_min = 3e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, End_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        shuffle = True

        train_dataset = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train=True)
        val_dataset = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train=False)

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)

        # 记录eval的map曲线
        eval_callback = EvalCallback(model, input_shape, anchors, class_names, num_classes, val_lines, log_dir,
                                     Cuda, eval_flag=eval_flag, period=eval_period)

        # 开始模型训练
        for epoch in range(Init_Epoch, End_Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, criterion, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, End_Epoch, Cuda, False, None, save_period, save_dir)

        loss_history.writer.close()
