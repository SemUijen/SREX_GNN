import torch
from torch import Tensor
from typing import Tuple


def get_accuracy(prediction: Tensor, label: Tensor) -> Tuple[float, float, float]:
    binary_predict = torch.where(prediction > 0.5, 1, 0)
    binary_label = torch.where(label > 0.5, 1, 0)
    equality = torch.eq(binary_predict, binary_label)

    total_accuracy = len(torch.where(equality == True)[0]) / len(prediction)

    pos_pred = equality[binary_predict.nonzero()]
    if len(pos_pred) == 0:
        pos_acc = 1
    else:
        pos_acc = len(torch.where(pos_pred == True)[0]) / len(pos_pred)

    neg_pred = equality[torch.where(binary_predict == 0)[0]]
    if len(neg_pred) == 0:
        false_neg = 0
    else:
        false_neg = 1 - len(torch.where(neg_pred == True)[0]) / len(neg_pred)

    return total_accuracy, pos_acc, false_neg


def get_accuracy_adjusted(prediction: Tensor, label: Tensor) -> Tuple[float, float, float]:
    binary_predict = torch.where(prediction > 0.9, 1, 0)
    binary_label = torch.where(label > 0.9, 1, 0)
    equality = torch.eq(binary_predict, binary_label)

    pos_pred = equality[binary_predict.nonzero()]
    if len(pos_pred) == 0:
        pos_acc_adj = 1
    else:
        pos_acc_adj = len(torch.where(pos_pred == True)[0]) / len(pos_pred)

    return pos_acc_adj


class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor):
        tensor.float()
        dist = (tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0])
        print(dist)
        dist[dist == 0.] = 1.
        scale = 1.0 / dist
        tensor.mul_(scale)  # .sub_(tensor.min(dim=0, keepdim=True)[0])
        return tensor
