import torch
from torch import Tensor


class Weights:
    def __init__(self, weight_type: bool):
        self.weight_type = weight_type

    def __call__(self, label, prediction, acc_score, device):
        if self.weight_type is None:
            return torch.ones(label.shape, device=device)

        if self.weight_type == "strong":
            return self.get_weight_strong(label, acc_score, device)

        if self.weight_type == "confuse":
            return self.conf_matrix_weight(label, prediction, device)

    def get_weight_strong(self, label, acc_score, device) -> Tensor:
        tot = len(label)
        pos = round((acc_score * len(label)).item())
        neg = tot - pos

        if pos == 0 or neg == 0:
            return torch.ones(label.shape, device=device)

        pos_w = tot / (2 * pos)
        neg_w = tot / (2 * neg)

        weights = torch.where(label > 0, pos_w, neg_w).to(device)
        return weights

    def conf_matrix_weight(self, label, prediction, device) -> Tensor:
        binary_predict = torch.where(prediction > 0.5, 1, 0)
        binary_label = torch.where(label > 0, 1, 0)
        equality = torch.eq(binary_predict, binary_label)

        weight = torch.ones(label.shape, device=device)
        pos_pred = equality[binary_predict.nonzero()]
        if len(pos_pred) == 0:
            pos_acc = 1
        else:
            weight[torch.where(pos_pred == False)[0]] = 1.2

        weight[torch.where(label > 0)] = 5
        return weight
