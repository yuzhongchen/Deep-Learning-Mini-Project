import argparse
import logging
import os.path

import torch
from tqdm import tqdm
import dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import extra_loss
import random

import resnet
from scipy.interpolate import make_interp_spline

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default='./result/depths_110+C+SGD')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--milestones', type=str, default='[120, 240]')
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--block_type', type=str, default='basic')
parser.add_argument('--depths', type=int, default=110)
parser.add_argument('--base_channels', type=int, default=16)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--input_shape', type=tuple, default=[1, 3, 32, 32])
parser.add_argument('--seed', type=int, default=17)
opts = parser.parse_args()


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    output_path = os.path.join(opts.result_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = '%s/train.txt' % output_path
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    plain_formatter = logging.Formatter(
        '[%(asctime)s]: %(message)s', datefmt='%m/%d %H:%M:%S'
    )
    handler.setFormatter(plain_formatter)
    logger.addHandler(handler)
    logger.info(opts)

    # set random seed
    seed = opts.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = dataset.CIFARDataset(split='train_50000')
    test_data = dataset.CIFARDataset(split='test')

    model = resnet.Network(opts)
    model.to(device)
    train_iter = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
                            pin_memory=True, drop_last=True)
    test_iter = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                           pin_memory=True)

    criterion_CE = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay,
        nesterov=True
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=opts.milestones,
        gamma=opts.lr_decay
    )
    train_loss_all = []
    train_accur_all = []
    test_loss_all = []
    test_accur_all = []
    best_epoch = 0
    best_accuracy = 0.0

    for epoch in range(opts.epochs):
        train_loss = 0
        train_num = 0
        train_accuracy = 0.0
        test_accuracy = 0.0
        model.train()
        train_bar = tqdm(train_iter)
        for step, (data, label) in enumerate(train_bar):
            label = np.array(label).astype(int)
            label = torch.from_numpy(label)
            one_hot_label = to_onehot(label, num_classes=10)
            one_hot_label = torch.from_numpy(one_hot_label)

            optimizer.zero_grad()
            outputs = model(data.to(device))

            loss_CE = criterion_CE(outputs, label.to(device).long())
            loss_Focal = calc_focal_loss(outputs, one_hot_label.to(device).long())
            loss_Dice = calc_dice_loss(outputs, label.to(device).long())
            loss_train = loss_CE
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, dim=1)
            loss_ = loss_train.item()
            correct_ = preds.eq(label.to(device)).sum().item()
            train_loss += abs(loss_) * data.size(0)
            train_accuracy += correct_
            train_num += data.size(0)

        print('epoch: %d, train_loss: %f, train_accuracy: %.4f' % (epoch + 1, train_loss / train_num,
                                                                   train_accuracy / train_num))
        logger.info('epoch: %d, train_loss: %f, train_accuracy: %.4f' % (epoch + 1, train_loss / train_num,
                                                                         train_accuracy / train_num))
        train_loss_all.append(train_loss / train_num)
        train_accur_all.append(train_accuracy / train_num)

        model.eval()
        test_num = 0
        test_loss = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_iter)
            for data in test_bar:
                img, test_label = data
                test_label = np.array(test_label).astype(int)
                test_label = torch.from_numpy(test_label)
                one_hot_test_label = to_onehot(test_label, num_classes=10)
                one_hot_test_label = torch.from_numpy(one_hot_test_label)
                pre_outputs = model(img.to(device))

                loss_t_CE = criterion_CE(pre_outputs, test_label.to(device).long())
                loss_t_Focal = calc_focal_loss(pre_outputs, one_hot_test_label.to(device).long())
                loss_t_Dice = calc_dice_loss(pre_outputs, test_label.to(device).long())
                loss_test = loss_t_CE
                _, pred_label = torch.max(pre_outputs, 1)
                test_loss += abs(loss_test.item()) * img.size(0)
                accuracy = pred_label.eq(test_label.to(device)).sum().item()
                test_accuracy += accuracy
                test_num += img.size(0)

        current_t_accuracy = test_accuracy / test_num
        test_loss_all.append(test_loss / test_num)
        test_accur_all.append(current_t_accuracy)
        if current_t_accuracy > best_accuracy:
            if not os.path.exists('%s/model' % output_path):
                os.makedirs('%s/model' % output_path)
            torch.save(model.state_dict(), '%s/model/The best model ResNet.pth' % output_path)
            best_accuracy = current_t_accuracy
            best_epoch = epoch + 1
        print('Epoch Test: %d, Current test accuracy: %.4f, Best test accuracy in %d: %.4f' % (epoch + 1,
                                                                                               current_t_accuracy,
                                                                                               best_epoch,
                                                                                               best_accuracy,
                                                                                               ))
        logger.info('Epoch Test: %d, Current test accuracy: %.4f, Best test accuracy in %d: %.4f' % (epoch + 1,
                                                                                                     current_t_accuracy,
                                                                                                     best_epoch,
                                                                                                     best_accuracy))

    return train_loss_all, train_accur_all, test_loss_all, test_accur_all


def to_onehot(label, num_classes):
    one_hot = np.zeros((label.shape[0], num_classes))
    one_hot[np.arange(label.shape[0]), label] = 1
    return one_hot


def calc_focal_loss(pred, target):
    loss = extra_loss.focal_loss(pred, target)
    return loss


def calc_dice_loss(pred, target):
    loss = extra_loss.dice_loss(pred, target)
    return loss


def plot_result(train_loss_all, train_accur_all, test_loss_all, test_accur_all):
    epochs = len(train_loss_all)
    x = np.linspace(1, epochs, epochs)  # 生成 x 轴数据

    # 平滑处理 train_loss_all 和 test_loss_all
    train_loss_all_smooth = make_interp_spline(x, train_loss_all)(x)
    test_loss_all_smooth = make_interp_spline(x, test_loss_all)(x)

    # 平滑处理 train_accur_all 和 test_accur_all
    train_accur_all_smooth = make_interp_spline(x, train_accur_all)(x)
    test_accur_all_smooth = make_interp_spline(x, test_accur_all)(x)

    # 绘制平滑曲线
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss_all_smooth, "r-", label='Train loss')
    plt.plot(x, test_loss_all_smooth, 'b-', label='Test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(x, train_accur_all_smooth, 'r-', label='Train accur')
    plt.plot(x, test_accur_all_smooth, 'b-', label='Test accur')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('%s/result.png' % opts.result_path)
    plt.show()


if __name__ == '__main__':
    train_loss, train_acc, test_loss, test_acc = main()
    plot_result(train_loss, train_acc, test_loss, test_acc)
