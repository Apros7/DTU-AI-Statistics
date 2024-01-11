from automation import Tester
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier 

def KNN(x_train, x_test, y_train, y_test, func_var):

    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    K=func_var
    dist=1
    metric = 'minkowski'
    metric_params = {}
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, metric=metric, metric_params=metric_params)
    knclassifier.fit(x_train, y_train)
    y_pred = knclassifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
  

if __name__ == "__main__":
    lambda_to_test = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    tester = Tester(function_to_test = KNN, final_test = False, k = 14, vars_to_test=lambda_to_test)