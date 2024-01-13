from automation import Tester
from logistic_regression import LogReg
from class_tree import ClassTree
from KNN_tester import KNN

tester = Tester(final_test = False, _run_tests = False)
data_x, data_y = tester.get_data()
x_columns, y_columns = tester.get_data_columns()
data_fold_indexes = tester.get_data_folds()

def run_forward_selection():

