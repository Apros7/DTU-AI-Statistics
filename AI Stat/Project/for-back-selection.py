from automation import Tester
from logistic_regression import LogReg
from class_tree import ClassTree
from KNN_tester import KNN

import numpy as np

class Selector():
    def __init__(self, ml_function, ml_func_parameter, data_x, data_y, x_columns, data_fold_indexes) -> None:
        self.ml_func, self.ml_parameter, self.data_x, self.data_y, self.x_cols, self.data_folds = ml_function, ml_func_parameter, data_x, data_y, x_columns, data_fold_indexes

    def _get_all_forward_combs(self, data_x_implemented, data_x_left, columns_let_to_choose_from):
        combs = []
        for i in range(len(columns_let_to_choose_from)):
            columns_left = columns_let_to_choose_from[:i] + columns_let_to_choose_from[i+1:]
            column_added = columns_let_to_choose_from[i]
            data_to_add = data_x_left[:, i]
            data_left = data_x_left[:, np.arange(data_x_left.shape[1]) != i]
            new_data = data_x_implemented + data_to_add[:, np.newaxis] if data_x_implemented is not None else data_to_add[:, np.newaxis]
            combs.append((new_data, data_left, columns_left, column_added))
        return combs

    def forward_selection(self):
        columns_implemented = self._run_forward_selection()
        print(f"Forward Selection suggest: {columns_implemented}")

    def _run_forward_selection(self):
        data_implemented = None
        data_left = self.data_x
        columns_left = self.x_cols
        columns_implemented = []
        accuracy_to_beat = 0
        for _ in range(len(self.x_cols)):
            print(f"\n ROUND {_}")
            combs = self._get_all_forward_combs(data_implemented, data_left, columns_left)
            accuracy, data_implemented_sel, data_left_sel, columns_left_sel, column_implemented = self._run_one_forward_selection(combs, data_implemented, data_left, columns_left)
            if accuracy <= accuracy_to_beat: return columns_implemented
            data_implemented, data_left, columns_left = data_implemented_sel, data_left_sel, columns_left_sel
            columns_implemented.append(column_implemented)
            accuracy_to_beat = accuracy
        return columns_implemented

    def _run_cross_validation(self, data_x):
        accuracies = []
        for fold_train_indexes, fold_test_indexes in self.data_folds:
            fold_train_x, fold_train_y, fold_test_x, fold_test_y = data_x[fold_train_indexes], self.data_y[fold_train_indexes], data_x[fold_test_indexes], self.data_y[fold_test_indexes]
            accuracies.append(self.ml_func(fold_train_x, fold_test_x, fold_train_y, fold_test_y, self.ml_parameter))
        return np.mean(accuracies)

    def _run_one_forward_selection(self, combs, data_implemented, data_left, columns_left):
        accuracies = []
        for (new_data, data_left, columns_left, column_added) in combs:
            accuracies.append((self._run_cross_validation(new_data), data_implemented, data_left, columns_left, column_added))
        print(*[[a[0], a[4]] for a in accuracies], sep="\n")
        return max(accuracies, key=lambda x: x[0])

if __name__ == "__main__":
    tester = Tester(final_test = True, _run_tests = False)
    data_x, data_y = tester.get_data()
    x_columns, y_columns = tester.get_data_columns()
    data_fold_indexes = tester.get_data_folds()
    selector = Selector(KNN, 8, data_x, data_y, x_columns, data_fold_indexes)
    selector.forward_selection()
        

        

