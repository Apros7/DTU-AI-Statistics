from automation import Tester

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib.image import imread
from sklearn.metrics import f1_score, accuracy_score

def ClassTree(x_train, x_test, y_train, y_test, func_var):

    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    criterion=func_var
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=100)
    dtc = dtc.fit(x_train,y_train)
    y_pred = dtc.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == "__main__":
    h_to_test = ["gini", "entropy", "log_loss"]
    tester = Tester(function_to_test = ClassTree, final_test = False, k = 14, vars_to_test=h_to_test)