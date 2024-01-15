from baseline import baseline
from class_tree import ClassTree
from KNN_tester import KNN
from ANN_tester import ann
from logistic_regression import LogReg
from automation import Tester

functions_to_compare = [baseline, ClassTree, KNN, LogReg, ann]
# functions_to_compare = [baseline, ann]

KNN_to_test = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
h_to_test = ["gini", "entropy", "log_loss"]
baseline_to_test = [0]
nn_to_test = [128]
lambda_to_test = [0.1, 1, 10, 100]

all_vars_to_test = [baseline_to_test, h_to_test, KNN_to_test, lambda_to_test, nn_to_test]
# all_vars_to_test = [baseline_to_test, nn_to_test]

tester = Tester(function_to_test = functions_to_compare, final_test = True, k = 14, cross_validation_level = 2, vars_to_test=all_vars_to_test)

accuracies = [[x for i, x in enumerate(sublist) if (i+2) % 4 == 0] for sublist in tester.all_results]

all_accs = []
for i in range(len(accuracies[0])):
    lst = []
    for l in range(len(accuracies)):
        lst.append(accuracies[l][i])
    all_accs.append(lst)

print(*all_accs, sep="\n")
        
