import torch
from torch import Tensor
from typing import Tuple


def get_accuracy(prediction: Tensor, label: Tensor) -> Tuple[float, float]:
    binary_predict = torch.where(prediction > 0.5, 1, 0)
    binary_label = torch.where(label > 0.5, 1, 0)
    equality = torch.eq(binary_predict, binary_label)

    total_accuracy = len(torch.where(equality == True)[0]) / len(prediction)

    pos_pred = equality[binary_predict.nonzero()]
    pos_acc = len(torch.where(pos_pred == True)[0]) / len(pos_pred)

    return total_accuracy, pos_acc


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
