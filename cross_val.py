import numpy as np
from sklearn.model_selection import KFold
from constants import atlas_switcher, correlation_switcher
from data import generate_examples_list, load_extra_info
from ml import calculate_statistical_features, separate_cross_validation, execute_logreg_model, execute_svm_model, execute_svm_linear_model, execute_knn_model, calculate_statistical_features_gcn, execute_nb_model, execute_rf_model, generate_stellar_graph, create_graph_classification_model, train_gcn_model, execute_gcn_model, select_best_features, init_results_map, process_results_map

#folder_list = ["dataset_all_clinics"]
folder_list = ["dataset4"]
for folder_path in folder_list:
    name = 'res-cross-val-' + folder_path + '.txt'
    res=open(name,'w')
    #define config params
    folds = KFold(n_splits=3)
    folder_path = folder_path + '/'
    atlas_id_list = [1,2,3,4,6]
    correlation_list = ["Pearson Correlation and Fisher Normalization", "Pearson Correlation"]
    thresh_list = [0.10, 0.20, 0.30, 0.50]
    binarize_coef_list = [0.1, 0.3, 0.5, 0.7]
    feature_selection_list = [0, 2]
    best_features= [True]
    best_features_quantity= [3,5]
    model_type = ["SVM","SVM_LINEAR","LR","RF","NB","KNN","GCN","ALL"]

    ### ATLAS CROSS VALIDATION PEARSON
    res.write('Validación cruzada { atlas_id, pearson }: \n')
    res.write('\n')
    for atlas_id in atlas_id_list:

        #fixed params
        correlation = "Pearson Correlation"
        thresh = 0.21
        binarize_coef = 0.3
        feature_selection_id = 0
        best_features_id = False
        best_features_quantity_id = 5
        model_type_id = "RF"

        #define params for GCN model
        epochs = 10  
        folds = 5 
        n_repeats = 3 

        #execute 
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
           
            results_map = init_results_map(y_test)

            #execute 
            if (model_type_id == "SVM"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
            
            elif (model_type_id == "SVM_LINEAR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "LR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "RF"):
                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "NB"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "KNN"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "ALL"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

            else:
                print("empty")

     
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

        exp_list = calculate_statistical_features_gcn(example_list)
        graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
        model, generator, test_accs = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
        res.write("Resultados atlas_id gcn={}:\n".format(atlas_id))
        res.write("\tAccuracy mean:{} ".format(np.mean(test_accs)*100))
        res.write("\tAccuracy std:{} ".format(np.std(test_accs)*100))
        res.write('\n')
  
    ### ATLAS CROSS VALIDATION FISHER
    res.write('Validación cruzada { atlas_id, fisher }: \n')
    res.write('\n')
    for atlas_id in atlas_id_list:

        #fixed params
        correlation = "Pearson Correlation and Fisher Normalization"
        thresh = 0.21
        binarize_coef = 0.3
        feature_selection_id = 0
        best_features_id = False
        best_features_quantity_id = 5
        model_type_id = "RF"

        #define params for GCN model
        epochs = 10  
        folds = 5 
        n_repeats = 3 

        #execute 
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
           
            results_map = init_results_map(y_test)

            #execute 
            if (model_type_id == "SVM"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
            
            elif (model_type_id == "SVM_LINEAR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "LR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "RF"):
                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "NB"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "KNN"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "ALL"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

            else:
                print("empty")

     
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

        exp_list = calculate_statistical_features_gcn(example_list)
        graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
        model, generator, test_accs = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
        res.write("Resultados atlas_id gcn={}:\n".format(atlas_id))
        res.write("\tAccuracy mean:{} ".format(np.mean(test_accs)*100))
        res.write("\tAccuracy std:{} ".format(np.std(test_accs)*100))
        res.write('\n')

    ### BINARIZE PEARSON COEF CROSS VALIDATION
    res.write('Validación cruzada { binarize_coef }: \n')
    res.write('\n')
    for binarize_coef in binarize_coef_list:

        #fixed params
        correlation = "Pearson Correlation"
        atlas_id = 1
        thresh = 0.21
        feature_selection_id = 0
        best_features_id = False
        best_features_quantity_id = 5
        model_type_id = "RF"

        #define params for GCN model
        epochs = 10 
        folds = 5 
        n_repeats = 3 

        #execute 
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
           
            results_map = init_results_map(y_test)

            #execute 
            if (model_type_id == "SVM"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
            
            elif (model_type_id == "SVM_LINEAR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "LR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "RF"):
                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "NB"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "KNN"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "ALL"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

            else:
                print("empty")

     
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

        exp_list = calculate_statistical_features_gcn(example_list)
        graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
        model, generator, test_accs = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
        res.write("Resultados binarize_coef gcn={}:\n".format(binarize_coef))
        res.write("\tAccuracy mean:{} ".format(np.mean(test_accs)*100))
        res.write("\tAccuracy std:{} ".format(np.std(test_accs)*100))
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
        best_features_id = False
        best_features_quantity_id = 5
        model_type_id = "RF"

        #define params for GCN model
        epochs = 10
        folds = 5 
        n_repeats = 3 

        #execute 
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
           
            results_map = init_results_map(y_test)

            #execute 
            if (model_type_id == "SVM"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
            
            elif (model_type_id == "SVM_LINEAR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "LR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "RF"):
                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "NB"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "KNN"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "ALL"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

            else:
                print("empty")

     
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

        exp_list = calculate_statistical_features_gcn(example_list)
        graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
        model, generator, test_accs = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
        res.write("Resultados thresh gcn={}:\n".format(thresh))
        res.write("\tAccuracy mean:{} ".format(np.mean(test_accs)*100))
        res.write("\tAccuracy std:{} ".format(np.std(test_accs)*100))
        res.write('\n')

    ### FEATURES CROSS VALIDATION
    res.write('Validación cruzada { feature_selection_id }: \n')
    res.write('\n')
    for feature_selection_id in feature_selection_list:

        #fixed params
        atlas_id = 1
        correlation = "Pearson Correlation"
        thresh = 0.21
        binarize_coef = 0.3
        best_features_id = False
        best_features_quantity_id = 5
        model_type_id = "RF"    

        #define params for GCN model
        epochs = 15  
        folds = 5 
        n_repeats = 3 

        #execute 
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
           
            results_map = init_results_map(y_test)

            #execute 
            if (model_type_id == "SVM"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
            
            elif (model_type_id == "SVM_LINEAR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "LR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "RF"):
                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "NB"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "KNN"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "ALL"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

            else:
                print("empty")

     
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
   
    ### BEST FEATURES CROSS VALIDATION
    res.write('Validación cruzada { best features }: \n')
    res.write('\n')
    for best_features_quantity_id in best_features_quantity:

        #fixed params
        atlas_id = 1
        correlation = "Pearson Correlation"
        thresh = 0.21
        binarize_coef = 0.3
        feature_selection_id = 0
        best_features_id = True
        model_type_id = "RF"

        #define params for GCN model
        epochs = 15  
        folds = 5 
        n_repeats = 3 

        #execute 
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
           
            results_map = init_results_map(y_test)

            #execute 
            if (model_type_id == "SVM"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
            
            elif (model_type_id == "SVM_LINEAR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "LR"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "RF"):
                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "NB"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "KNN"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

            elif (model_type_id == "ALL"):

                if (best_features_id):
                    X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

            else:
                print("empty")

     
            results_accuracy.append(acc)
            results_precision.append(pre)
            results_recall.append(rec)
            results_f1.append(f1)

        res_accuracy = np.average(results_accuracy)
        res_precision = np.average(results_precision)
        res_recall = np.average(results_recall)
        res_f1 = np.average(results_f1)

        res.write("Resultados best features quantity={}:\n".format(best_features_quantity_id))
        res.write("\tAccuracy:{} ".format(res_accuracy))
        res.write("\tPrecision:{} ".format(res_precision))
        res.write("\tRecall:{} ".format(res_recall))
        res.write("\tF1:{} \n".format(res_f1))
        res.write('\n')
  
    ### MODEL CROSS VALIDATION
    res.write('Validación cruzada { model }: \n')
    res.write('\n')
    for model_type_id in model_type:

        #fixed params
        correlation = "Pearson Correlation"
        atlas_id = 1
        thresh = 0.21
        binarize_coef = 0.3
        feature_selection_id = 0
        best_features_id = False
        best_features_quantity_id = 5

        #execute 
        extra_info = load_extra_info()
        example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
        
        if (model_type_id == "GCN"):

            #define params for GCN model
            epochs_list = [20, 30, 40, 50]  
            folds = 5 
            n_repeats = 3 
            
            for epochs in epochs_list:
                exp_list = calculate_statistical_features_gcn(example_list)
                graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
                model, generator, test_accs = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
                res.write("Resultados model_type={}, ".format(model_type_id))
                res.write("epochs={}:\n".format(epochs))
                res.write("\tAccuracy mean:{} ".format(np.mean(test_accs)*100))
                res.write("\tAccuracy std:{} ".format(np.std(test_accs)*100))
                res.write('\n')

        else:
        
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
            
                results_map = init_results_map(y_test)

                #execute 
                if (model_type_id == "SVM"):

                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                    acc, pre, rec, f1, results_map = execute_svm_model(X_train, y_train, X_test, y_test, results_map)
                
                elif (model_type_id == "SVM_LINEAR"):

                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                    acc, pre, rec, f1, results_map = execute_svm_linear_model(X_train, y_train, X_test, y_test, results_map)

                elif (model_type_id == "LR"):

                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id) 
                    acc, pre, rec, f1, results_map = execute_logreg_model(X_train, y_train, X_test, y_test, results_map)

                elif (model_type_id == "RF"):
                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                    acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)

                elif (model_type_id == "NB"):

                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                    acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)

                elif (model_type_id == "KNN"):

                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                    acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)

                elif (model_type_id == "ALL"):

                    if (best_features_id):
                        X_train, X_test = select_best_features(X_train, X_test, y_train, y_test, best_features_quantity_id)      
                    acc, pre, rec, f1, results_map = execute_knn_model(X_train, y_train, X_test, y_test, results_map)
                    acc, pre, rec, f1, results_map = execute_nb_model(X_train, y_train, X_test, y_test, results_map)
                    acc, pre, rec, f1, results_map = execute_rf_model(X_train, y_train, X_test, y_test, results_map)
                    acc, pre, rec, f1 = process_results_map(results_map, "Traditional Classifiers", y_test)

                else:
                    print("empty")

        
                results_accuracy.append(acc)
                results_precision.append(pre)
                results_recall.append(rec)
                results_f1.append(f1)

            res_accuracy = np.average(results_accuracy)
            res_precision = np.average(results_precision)
            res_recall = np.average(results_recall)
            res_f1 = np.average(results_f1)

            res.write("Resultados model_type={}:\n".format(model_type_id))
            res.write("\tAccuracy:{} ".format(res_accuracy))
            res.write("\tPrecision:{} ".format(res_precision))
            res.write("\tRecall:{} ".format(res_recall))
            res.write("\tF1:{} \n".format(res_f1))
            res.write('\n')
    









