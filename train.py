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

import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default='./result/04031900')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--milestones', type=str, default='[80, 120]')
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--block_type', type=str, default='bottleneck')
parser.add_argument('--depths', type=int, default=110)
parser.add_argument('--base_channels', type=int, default=16)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--input_shape', type=tuple, default=[1, 3, 32, 32])
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

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = dataset.CIFARDataset(split='train')
    val_data = dataset.CIFARDataset(split='val')
    test_data = dataset.CIFARDataset(split='test')

    model = resnet.Network(opts)
    model.to(device)
    train_iter = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_iter = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=opts.milestones,
        gamma=opts.lr_decay
    )
    train_loss_all = []
    train_accur_all = []
    test_loss_all = []
    test_accur_all = []

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
            outputs = model(data.to(device))

            optimizer.zero_grad()
            loss_train = criterion(outputs, label.to(device))
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, dim=1)
            loss_ = loss_train.item()
            correct_ = preds.eq(label.to(device)).sum().item()
            train_loss += abs(loss_) * data.size(0)
            train_accuracy += correct_
            train_num += data.size(0)

        print('epoch: %d, train_loss: %f, train_accuracy: %.2f' % (epoch + 1, train_loss / train_num,
                                                                 train_accuracy / train_num))
        logger.info('epoch: %d, train_loss: %f, train_accuracy: %.2f' % (epoch + 1, train_loss / train_num,
                                                                 train_accuracy / train_num))
        train_loss_all.append(train_loss / train_num)
        train_accur_all.append(train_accuracy / train_num)
        if (epoch + 1) % 10 == 0:
            if not os.path.exists('%s/model' % output_path):
                os.makedirs('%s/model' % output_path)
            torch.save(model.state_dict(), '%s/model/ResNet_%s.pth' % (output_path, epoch + 1))

        model.eval()
        test_num = 0
        test_loss = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_iter)
            for data in test_bar:
                img, test_label = data
                test_label = np.array(test_label).astype(int)
                test_label = torch.from_numpy(test_label)
                pre_outputs = model(img.to(device))

                loss_test = criterion(pre_outputs, test_label.to(device))

                _, pred_label = torch.max(pre_outputs, 1)
                test_loss += abs(loss_test.item()) * img.size(0)
                accuracy = pred_label.eq(test_label.to(device)).sum().item()
                test_accuracy += accuracy
                test_num += img.size(0)

        print('Epoch Test: %d, Test_accuracy: %.2f' % (epoch + 1, test_accuracy / test_num))
        logger.info('Epoch Test: %d, Test_accuracy: %.2f' % (epoch + 1, test_accuracy / test_num))
        test_loss_all.append(test_loss / test_num)
        test_accur_all.append(test_accuracy / test_num)

    return train_loss_all, train_accur_all, test_loss_all, test_accur_all


def plot_result(train_loss_all, train_accur_all, test_loss_all, test_accur_all):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, opts.epochs + 1), train_loss_all,
             "ro-", label='Train loss')
    plt.plot(range(1, opts.epochs + 1), test_loss_all,
             'bo-', label='Test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, opts.epochs + 1), train_accur_all,
             'ro-', label='Train accur')
    plt.plot(range(1, opts.epochs + 1), test_accur_all,
             'bo-', label='Test accur')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('%s/result.png' % opts.result_path)
    plt.show()


if __name__ == '__main__':
    train_loss, train_acc, test_loss, test_acc = main()
    plot_result(train_loss, train_acc, test_loss, test_acc)
