import numpy as np
from sklearn.model_selection import KFold
from constants import atlas_switcher, correlation_switcher
from data import generate_examples_list, load_extra_info
from ml import calculate_statistical_features, separate_cross_validation, execute_logreg_model, execute_svm_model

folder_list = ["dataset1", "dataset2", "dataset3", "datasets"]
for folder_path in folder_list:
    name = 'res-cross-validation-' + folder_path + '.txt'
    res=open(name,'w')
    #define config params
    folds = KFold(n_splits=3)
    folder_path = folder_path + '/'
    atlas_id_list = [1]
    correlation_list = ["Pearson Correlation and Fisher Normalization", "Pearson Correlation"]
    thresh_list = [0.10, 0.20, 0.30, 0.50]
    binarize_coef_list = [0.1, 0.3, 0.5, 0.7]
    feature_selection_list = [0, 1, 2, 3]

    ### ATLAS CROSS VALIDATION FISHER
    res.write('Validación cruzada { atlas_id, fisher }: \n')
    res.write('\n')
    for atlas_id in atlas_id_list:

        #fixed params
        correlation = "Pearson Correlation and Fisher Normalization"
        thresh = 0.21
        binarize_coef = 0.3
        feature_selection_id = 0

        #preprocess
        extra_info = load_extra_info()
        example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
        exp_list = calculate_statistical_features(example_list)
        train, label = separate_cross_validation(exp_list, feature_selection_id)
        X = np.array(train)
        y = np.array(label)
        kf = KFold(n_splits=3)
        kf.get_n_splits(X)
        KFold(n_splits=3, random_state=None, shuffle=False)

        #results arrays
        results_accuracy=[]
        results_recall=[]
        results_precision=[]
        results_f1=[]

        #cross validation
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

        res.write("Resultados atlas_id={}:\n".format(atlas_id))
        res.write("\tAccuracy:{} ".format(res_accuracy))
        res.write("\tPrecision:{} ".format(res_precision))
        res.write("\tRecall:{} ".format(res_recall))
        res.write("\tF1:{} \n".format(res_f1))
        res.write('\n')

    ### ATLAS CROSS VALIDATION PEARSON
    res.write('Validación cruzada { atlas_id, pearson }: \n')
    res.write('\n')
    for atlas_id in atlas_id_list:

        #fixed params
        correlation = "Pearson Correlation"
        thresh = 0.21
        binarize_coef = 0.3
        feature_selection_id = 0

        #preprocess
        extra_info = load_extra_info()
        example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
        exp_list = calculate_statistical_features(example_list)
        train, label = separate_cross_validation(exp_list, feature_selection_id)
        X = np.array(train)
        y = np.array(label)
        kf = KFold(n_splits=3)
        kf.get_n_splits(X)
        KFold(n_splits=3, random_state=None, shuffle=False)

        #results arrays
        results_accuracy=[]
        results_recall=[]
        results_precision=[]
        results_f1=[]

        #cross validation
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

        res.write("Resultados atlas_id={}:\n".format(atlas_id))
        res.write("\tAccuracy:{} ".format(res_accuracy))
        res.write("\tPrecision:{} ".format(res_precision))
        res.write("\tRecall:{} ".format(res_recall))
        res.write("\tF1:{} \n".format(res_f1))
        res.write('\n')


    ### BINARIZE PEARSON COEF CROSS VALIDATION
    res.write('Validación cruzada { binarize_coef }: \n')
    res.write('\n')
    for binarize_coef in binarize_coef_list:

        #fixed params
        atlas_id = 1
        correlation = "Pearson Correlation"
        thresh = 0.21
        feature_selection_id = 0

        #preprocess
        extra_info = load_extra_info()
        example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
        exp_list = calculate_statistical_features(example_list)
        train, label = separate_cross_validation(exp_list, feature_selection_id)
        X = np.array(train)
        y = np.array(label)
        kf = KFold(n_splits=3)
        kf.get_n_splits(X)
        KFold(n_splits=3, random_state=None, shuffle=False)

        #results arrays
        results_accuracy=[]
        results_recall=[]
        results_precision=[]
        results_f1=[]

        #cross validation
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

        res.write("Resultados binarize_coef={}:\n".format(binarize_coef))
        res.write("\tAccuracy:{} ".format(res_accuracy))
        res.write("\tPrecision:{} ".format(res_precision))
        res.write("\tRecall:{} ".format(res_recall))
        res.write("\tF1:{} \n".format(res_f1))
        res.write('\n')


    ### THRESH CROSS VALIDATION
    res.write('Validación cruzada { thresh }: \n')
    res.write('\n')
    for thresh in thresh_list:

        #fixed params
        atlas_id = 1
        correlation = "Pearson Correlation and Fisher Normalization"
        binarize_coef = 0.3
        feature_selection_id = 0

        #preprocess
        extra_info = load_extra_info()
        example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
        exp_list = calculate_statistical_features(example_list)
        train, label = separate_cross_validation(exp_list, feature_selection_id)
        X = np.array(train)
        y = np.array(label)
        kf = KFold(n_splits=3)
        kf.get_n_splits(X)
        KFold(n_splits=3, random_state=None, shuffle=False)

        #results arrays
        results_accuracy=[]
        results_recall=[]
        results_precision=[]
        results_f1=[]

        #cross validation
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

        res.write("Resultados thresh={}:\n".format(thresh))
        res.write("\tAccuracy:{} ".format(res_accuracy))
        res.write("\tPrecision:{} ".format(res_precision))
        res.write("\tRecall:{} ".format(res_recall))
        res.write("\tF1:{} \n".format(res_f1))
        res.write('\n')
    
     ### FEATUES CROSS VALIDATION
    res.write('Validación cruzada { feature_selection_id }: \n')
    res.write('\n')
    for feature_selection_id in feature_selection_list:

        #fixed params
        atlas_id = 1
        correlation = "Pearson Correlation"
        thresh = 0.21

        #preprocess
        extra_info = load_extra_info()
        example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
        exp_list = calculate_statistical_features(example_list)
        train, label = separate_cross_validation(exp_list, feature_selection_id)
        X = np.array(train)
        y = np.array(label)
        kf = KFold(n_splits=3)
        kf.get_n_splits(X)
        KFold(n_splits=3, random_state=None, shuffle=False)

        #results arrays
        results_accuracy=[]
        results_recall=[]
        results_precision=[]
        results_f1=[]

        #cross validation
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

        res.write("Resultados feature_selection_id={}:\n".format(feature_selection_id))
        res.write("\tAccuracy:{} ".format(res_accuracy))
        res.write("\tPrecision:{} ".format(res_precision))
        res.write("\tRecall:{} ".format(res_recall))
        res.write("\tF1:{} \n".format(res_f1))
        res.write('\n')