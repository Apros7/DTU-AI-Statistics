from automation import Tester

tester = Tester(final_test = False, _run_tests = False)
data_x, data_y = tester.get_data()
data_fold_indexes = tester.get_data_folds()

