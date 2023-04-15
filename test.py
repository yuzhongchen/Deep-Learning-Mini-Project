import argparse
import logging
import os.path

import torch
import resnet
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default='./result/depths_56+CFD+Adam')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--block_type', type=str, default='basic')
parser.add_argument('--depths', type=int, default=56)
parser.add_argument('--base_channels', type=int, default=16)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--input_shape', type=tuple, default=[1, 3, 32, 32])
opts = parser.parse_args()

def test():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    output_path = os.path.join(opts.result_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = '%s/test.txt' % output_path
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    plain_formatter = logging.Formatter(
        '[%(asctime)s]: %(message)s', datefmt='%m/%d %H:%M:%S'
    )
    handler.setFormatter(plain_formatter)
    logger.addHandler(handler)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = dataset.CIFARDataset(split='test')

    model = resnet.Network(opts)
    model.to(device)
    test_iter = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=True)

    pretrain_model_path = '%s/model/The best model ResNet.pth' % opts.result_path
    model.load_state_dict(torch.load(pretrain_model_path))

    model.eval()
    test_num = 0
    test_accuracy = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_iter)
        for data in test_bar:
            img, test_label = data
            test_label = np.array(test_label).astype(int)
            test_label = torch.from_numpy(test_label)
            pre_outputs = model(img.to(device))

            _, pred_label = torch.max(pre_outputs, 1)
            accuracy = pred_label.eq(test_label.to(device)).sum().item()
            test_accuracy += accuracy
            test_num += img.size(0)
    print('Test_accuracy: %.4f' % (test_accuracy / test_num))
    logger.info('Test_accuracy: %.4f' % (test_accuracy / test_num))


if __name__ == '__main__':
    test()
