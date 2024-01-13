from automation import Tester
from logistic_regression import LogReg
from class_tree import ClassTree
from KNN_tester import KNN

tester = Tester(final_test = False, _run_tests = False)
data_x, data_y = tester.get_data()
x_columns, y_columns = tester.get_data_columns()
data_fold_indexes = tester.get_data_folds()

class Selector():
    def __init__(self, ml_function, data_x, data_y, x_columns, data_fold_indexes) -> None:
        self.ml_func, self.data_x, self.data_y, self.x_cols, self.data_folds = ml_function, data_x, data_y, x_columns, data_fold_indexes

    def _run_one_fold(self, x_train, y_train, x_test, y_test): return self.ml_func(x_train, y_train, x_test, y_test)

    def _get_all_forward_combs(self, data_x_implemented, data_x_left, columns_let_to_choose_from):
        


    def _run_one_forward_selection(self):
        
            results_this_fold = []
            fold_train_x, fold_train_y, fold_test_x, fold_test_y = self.data_x[fold_train_indexes], self.data_y[fold_train_indexes], self.data_x[fold_test_indexes], self.data_y[fold_test_indexes]


    fold_train_x, fold_train_y, fold_test_x, fold_test_y = self.data_x[fold_train_indexes], self.data_y[fold_train_indexes], self.data_x[fold_test_indexes], self.data_y[fold_test_indexes]
        

