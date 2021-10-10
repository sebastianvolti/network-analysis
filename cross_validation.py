import numpy as np
from sklearn.model_selection import KFold
from constants import atlas_switcher, correlation_switcher
from data import generate_examples_list
from ml import calculate_statistical_features, separate_cross_validation, execute_logreg_model, execute_svm_model


res=open('res-cross-validation.txt','w')

#define params
folds = KFold(n_splits=3)
folder_path = "dataset/"
atlas_id_list = [1,2,3,4,6,8,9]
correlation_list = ["Pearson Correlation and Fisher Normalization", "Pearson Correlation"]
thresh_list = [0.21, 0.30, 0.50]
binarize_coef_list = [0.1, 0.3, 0.5, 0.7]

results=[]
results_accuracy=[]
results_recall=[]
results_precision=[]
results_f1=[]

atlas_id = 1
correlation = "Pearson Correlation and Fisher Normalization"
thresh = 0.21
binarize_coef = 0.3

#preprocess
example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef)
exp_list = calculate_statistical_features(example_list)
train, label = separate_cross_validation(exp_list)
X = np.array(train)
y = np.array(label)
kf = KFold(n_splits=3)
kf.get_n_splits(X)
KFold(n_splits=3, random_state=None, shuffle=False)

results_accuracy=[]
results_recall=[]
results_precision=[]
results_f1=[]

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #execute 
    acc, pre, rec, f1 = execute_svm_model(X_train, y_train, X_test, y_test)

    results_accuracy.append(acc)
    results_precision.append(pre)
    results_recall.append(rec)
    results_f1.append(f1)

res_accuracy = np.average(results_accuracy)
res_precision = np.average(results_precision)
res_recall = np.average(results_recall)
res_f1 = np.average(results_f1)

res.write('Resultados: \n')
res.write("\tAccuracy:{} ".format(res_accuracy))
res.write("\tPrecision:{} ".format(res_precision))
res.write("\tRecall:{} ".format(res_recall))
res.write("\tF1:{} ".format(res_f1))