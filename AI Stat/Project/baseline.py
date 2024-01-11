
from automation import Tester
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from collections import Counter
import torch

def baseline(x_train, x_test, y_train, y_test, func_var):
    y_train = torch.argmax(y_train, dim=1)
    most_common_class = max(set(y_train), key=y_train.tolist().count)
    y_pred = [most_common_class] * len(y_test)
    y_pred = torch.tensor([t.item() for t in y_pred])
    y_test = torch.argmax(y_test, dim=1)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
  
if __name__ == "__main__":
    tester = Tester(function_to_test = baseline, final_test = False, k = 14) # k = number of individuals

    h_to_test = ["gini", "entropy", "log_loss"]