import torch

predict = [-200, 0, 10, 200]
predict_T = torch.tensor(predict)
label = torch.tensor([-5, -8, 10, 30])

sigmoid = torch.sigmoid(predict_T)

print(sigmoid)
def get_accuracy(prediction, label):


    accuracy = 1
    return accuracy


get_accuracy(sigmoid, label)