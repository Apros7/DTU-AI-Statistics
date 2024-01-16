from automation import Tester
from logistic_regression import LogReg
from class_tree import ClassTree
from KNN_tester import KNN

import numpy as np

class Selector():
    def __init__(self, data_x, data_y, x_columns, data_fold_indexes, print_info) -> None:
        self.data_x, self.data_y, self.x_cols, self.data_folds, self.print_info = data_x, data_y, x_columns, data_fold_indexes, print_info

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

    def _get_all_backward_combs(self, data_x, columns_let_to_choose_from):
        combs = []
        for i in range(len(columns_let_to_choose_from)-1):
            all_other_columns = columns_let_to_choose_from[:i] + columns_let_to_choose_from[i+1:]
            column_deleted = columns_let_to_choose_from[i]
            data_left = data_x[:, np.arange(data_x.shape[1]) != i]
            combs.append((data_left, all_other_columns, column_deleted))
        return combs

    def run_forward_selection(self, ml_func, ml_parameter): return self.run_selection(ml_func, ml_parameter, backward = False)
    def run_backward_selection(self, ml_func, ml_parameter): return self.run_selection(ml_func, ml_parameter, backward = True)

    def run_selection(self, ml_func, ml_parameter, backward):
        self.ml_func, self.ml_parameter = ml_func, ml_parameter
        columns_implemented = self._run_backward_selection() if backward else self._run_forward_selection() 
        print(f"\n{'Backward' if backward else 'Forward'} Selection for {self.ml_func.__name__} suggest using: {columns_implemented}")
        return columns_implemented

    def _run_forward_selection(self):
        data_implemented = None
        data_left = self.data_x
        columns_left = self.x_cols
        columns_implemented = []
        accuracy_to_beat = 0
        for _ in range(len(self.x_cols)):
            if self.print_info: print(f"\n ROUND {_}: {accuracy_to_beat}")
            combs = self._get_all_forward_combs(data_implemented, data_left, columns_left)
            accuracy, data_implemented_sel, data_left_sel, columns_left_sel, column_implemented = self._run_one_forward_selection(combs, data_implemented, data_left, columns_left)
            if accuracy_to_beat >= accuracy: return columns_implemented
            data_implemented, data_left, columns_left = data_implemented_sel, data_left_sel, columns_left_sel
            columns_implemented.append(column_implemented)
            accuracy_to_beat = accuracy
        return columns_implemented

    def _run_backward_selection(self):
        data_implemented = self.data_x
        columns_implemented = self.x_cols
        accuracy_to_beat = self._run_cross_validation(self.data_x)
        for _ in range(len(self.x_cols)):
            if self.print_info: print(f"\n ROUND {_}: {accuracy_to_beat}")
            combs = self._get_all_backward_combs(data_implemented, columns_implemented)
            accuracy, new_data, all_other_columns, column_deleted = self._run_one_backward_selection(combs)
            if accuracy_to_beat >= accuracy: return columns_implemented
            accuracy_to_beat = accuracy
            data_implemented = new_data
            columns_implemented = all_other_columns
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
        if self.print_info: print(*[[a[0], a[4]] for a in accuracies], sep="\n")
        return max(accuracies, key=lambda x: x[0])

    def _run_one_backward_selection(self, combs):
        accuracies = []
        for (new_data, all_other_columns, column_deleted) in combs:
            accuracies.append((self._run_cross_validation(new_data), new_data, all_other_columns, column_deleted))
        if self.print_info: print(*[[a[0], a[2]] for a in accuracies], sep="\n")
        return max(accuracies, key=lambda x: x[0])

if __name__ == "__main__":
    tester = Tester(final_test = True, _run_tests = False)
    data_x, data_y = tester.get_data()
    x_columns, y_columns = tester.get_data_columns()
    data_fold_indexes = tester.get_data_folds()
    selector = Selector(data_x, data_y, x_columns, data_fold_indexes, print_info = True)
    selector.run_forward_selection(KNN, 8)
    selector.run_backward_selection(KNN, 8)
    selector.run_forward_selection(LogReg, 1)
    selector.run_backward_selection(LogReg, 1)
    selector.run_forward_selection(ClassTree, "gini")
    selector.run_backward_selection(ClassTree, "gini")
        

        

