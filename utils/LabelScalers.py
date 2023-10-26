import torch
from torch import Tensor
from typing import Tuple


# TODO: add scalers for Sigmoid: Scale plus and minues seperatly between -10 and 10 +/-

class SigmoidVectorizedScaler(object):
    """
    Transforms each tensor to the range [-10, 10].
    """

    def __init__(self, scale_range, device):
        self.scale_range = scale_range
        self.device = device
    def scale(self, input_tensor: Tensor, pos: bool) -> Tensor:
        tensor = input_tensor.clone()

        dist = (tensor.max().item() - tensor.min().item())
        if dist == 0.:
            return torch.zeros(tensor.shape, device=self.device)

        else:
            scale = self.scale_range / dist
            tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])

            return tensor if pos else tensor - self.scale_range

    def __call__(self, input_tensor: Tensor) -> Tensor:

        tensor = input_tensor.clone()

        tensor_neg_idx = torch.where(tensor <= 0)
        tensor_pos_idx = torch.where(tensor >= 0)

        if tensor_neg_idx[0].numel():
            scaled_neg = self.scale(tensor[tensor_neg_idx], False)
            tensor[tensor_neg_idx] = scaled_neg

        if tensor_pos_idx[0].numel():
            scaled_pos = self.scale(tensor[tensor_pos_idx], True)
            tensor[tensor_pos_idx] = scaled_pos

        return tensor


class SoftmaxVectorizedScaler(object):
    """
    Transforms each tensor to the range [0, 1].
    """

    def __call__(self, input_tensor: Tensor) -> Tensor:
        # TODO: Fix scale when there are no improvements LEADS TO PICKING ZERO as an improvement
        tensor = input_tensor.clone()

        dist = (tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0])
        if dist == 0.:
            return torch.zeros(tensor.shape)

        else:
            scale = 1.0 / dist
            tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
            return tensor





