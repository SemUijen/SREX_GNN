import torch
from torch import Tensor
from typing import Tuple


class Metrics():

    def __init__(self, name: str):

        self.name = name
        self.true_pos = 0
        self.false_pos = 0
        self.true_neg = 0
        self.false_neg = 0

        self.adjusted_acc = 0
        self.high_acc = 0

        self.tot_pos = 0
        self.tot_neg = 0
        self.num_couples = 0

        self.f1 = 0
        self.precision = 0
        self.recall = 0

        self.select_acc = 0
        self.select_high = 0

    def __call__(self, prediction: Tensor, label: Tensor):

        self.num_couples += 1
        self.get_confusion_matrix(prediction, label)
        self.get_accuracy_adjusted(prediction, label)
        self.get_high_acc(prediction, label)

        self.calculate_scores()

    def calculate_scores(self):
        if self.true_pos > 0:
            self.recall = self.true_pos / (self.true_pos + self.false_neg)
            self.precision = self.true_pos / (self.true_pos + self.false_pos)
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        if self.adjusted_acc > 0:
            self.select_acc = self.adjusted_acc / self.num_couples

        if self.high_acc > 0:
            self.select_high = self.high_acc/self.num_couples

    def get_accuracy_adjusted(self, prediction: Tensor, label: Tensor) -> float:

        max_prob = torch.where(prediction == prediction.max())[0]
        if prediction.max() > 0.5:
            if label[max_prob[0]] == 1:
                self.adjusted_acc += 1

        else:
            if label.max() == 0:
                self.adjusted_acc += 1

    def get_high_acc(self, prediction: Tensor, label: Tensor):
        max_prob = torch.where(prediction == prediction.max())[0]
        if label[max_prob[0]] == 1:
            self.high_acc += 1

    def get_confusion_matrix(self, prediction: Tensor, label: Tensor) -> None:
        binary_predict = torch.where(prediction > 0.5, 1, 0)
        binary_label = torch.where(label > 0.5, 1, 0)
        equality = torch.eq(binary_predict, binary_label)

        self.tot_pos += len(torch.where(binary_label == 1))
        self.tot_neg += len(torch.where(binary_label == 1))

        pos_pred = equality[binary_predict.nonzero()]
        if len(pos_pred) == 0:
            pos_acc = 1
        else:
            self.true_pos += len(torch.where(pos_pred == True)[0])
            self.false_pos += len(torch.where(pos_pred == False)[0])

        neg_pred = equality[torch.where(binary_predict == 0)[0]]
        if len(neg_pred) == 0:
            false_neg = 0
        else:
            self.false_neg += len(torch.where(neg_pred == False)[0])
            self.true_neg += len(torch.where(neg_pred == True)[0])

    def get_accuracy(self, prediction: Tensor, label: Tensor) -> Tuple[float, float, float]:
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

    def __str__(self):

        return (f"Metrics {self.name}: \n"
                f" TP = {self.true_pos} \n"
                f" FP = {self.false_pos} \n"
                f" TN = {self.true_neg} \n"
                f" FN = {self.false_neg} \n"
                f"\n"
                f" accuracy = {(self.true_pos + self.true_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg)} \n"
                f" precision = {self.precision} \n"
                f" recall = {self.recall} \n"
                f" F1_score = {self.f1} \n"
                f"\n"
                f" select_acc = {self.select_acc} \n"
                f" select_high = {self.select_high}")

