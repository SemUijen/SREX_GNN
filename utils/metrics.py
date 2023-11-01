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
        false_neg = len(torch.where(neg_pred == False)[0]) / len(neg_pred)
        true_neg = len(torch.where(neg_pred == True)[0]) / len(neg_pred)

    return total_accuracy, pos_acc, false_neg


def get_accuracy_adjusted(prediction: Tensor, label: Tensor) -> Tuple[float, float, float]:
    binary_predict = torch.where(prediction > 0.8, 1, 0)
    binary_label = torch.where(label > 0.8, 1, 0)
    equality = torch.eq(binary_predict, binary_label)

    pos_pred = equality[binary_predict.nonzero()]
    if len(pos_pred) == 0:
        pos_acc_adj = 1
    else:
        pos_acc_adj = len(torch.where(pos_pred == True)[0]) / len(pos_pred)

    return pos_acc_adj
