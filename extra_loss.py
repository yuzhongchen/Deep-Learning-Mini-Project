import torch
import torch.nn.functional as F


def focal_loss(y_pred, y_true_one_hot, gamma=2.0, alpha=0.25):
    """
    计算10分类Focal Loss

    参数：
    y_true_one_hot: 标签，shape为[N, 10]，每个元素为0或1
    y_pred: 模型输出，shape为[N, 10]，每个元素表示对应样本在10个类别上的得分
    gamma: Focal Loss中的gamma参数，默认为2.0
    alpha: Focal Loss中的alpha参数，默认为0.25

    返回值：
    计算得到的Focal Loss，shape为[]
    """
    # 计算模型输出经过softmax激活后的预测概率
    y_pred_softmax = torch.softmax(y_pred, dim=-1)

    # 计算预测概率与真实标签之间的差异
    pt = torch.sum(y_true_one_hot * y_pred_softmax, dim=-1)
    pt = torch.clamp(pt, min=1e-7, max=1-1e-7)

    # 计算Focal Loss
    fl = -alpha * (1 - pt) ** gamma * torch.log(pt)

    # 对所有样本的Focal Loss求和
    loss = fl.mean()

    return loss


def dice_loss(input, target, eps=1e-7):
    # 对类别维度进行求和
    num_classes = input.size(1)

    # 将预测输出转换成概率值
    input = F.softmax(input, dim=1)

    # 将目标张量转换为 one-hot 编码
    target_onehot = torch.eye(num_classes)[target].float().to(input.device)

    # 计算交集和并集
    intersection = torch.sum(input * target_onehot, dim=(0, 1))
    union = torch.sum(input, dim=(0, 1)) + torch.sum(target_onehot, dim=(0, 1))

    # 计算 Dice 系数
    dice = (2. * intersection + eps) / (union + eps)

    # 取 1 - Dice 作为损失函数值
    loss = 1. - dice.mean()

    return loss
