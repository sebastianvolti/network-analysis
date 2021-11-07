from data import generate_examples_list, load_extra_info
from ml import calculate_statistical_features, calculate_statistical_features_gcn, separate_training_validation, execute_logreg_model, execute_svm_model, execute_svm_linear_model, execute_knn_model, execute_nb_model, execute_rf_model, generate_stellar_graph, create_graph_classification_model, train_gcn_model, execute_gcn_model, select_best_features, init_results_map, process_results_map

def main():

    #model_type = "SVM"
    #model_type = "LR"
    #model_type = "RF"
    #odel_type = "GCN"
    model_type = "ALL"

    #define params for SVM or LR model
    #folder_path = "dataset_all_clinics/"
    folder_path = "dataset20/"
    atlas_id = 1
    #correlation = "Pearson Correlation and Fisher Normalization"
    correlation = "Pearson Correlation"
    thresh = 0.21
    binarize_coef = 0.3  
    feature_selection_id = 2 


    #define params for GCN model
    epochs = 5  
    folds = 5 
    n_repeats = 3 

    #execute 
    extra_info = load_extra_info()
    example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)
    if (model_type == "SVM" or model_type == "LR" or model_type == "RF"):
        exp_list = calculate_statistical_features(example_list)
        X, X_val, y, y_val = separate_training_validation(exp_list, feature_selection_id)
        #X, X_val = select_best_features(X, X_val, y, y_val, 5)
        results = execute_logreg_model(X, y, X_val, y_val)
        results = execute_svm_model(X, y, X_val, y_val)
        results = execute_rf_model(X, y, X_val, y_val)

    elif (model_type == "GCN"):
        exp_list = calculate_statistical_features_gcn(example_list)
        graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
        model, generator, acc = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
        execute_gcn_model(model, generator, len(exp_list))
    
    else:
        exp_list = calculate_statistical_features(example_list)
        X, X_val, y, y_val = separate_training_validation(exp_list, feature_selection_id)
        #X, X_val = select_best_features(X, X_val, y, y_val, 5)
        results_map = init_results_map(y_val)
        #acc, pre, rec, f1, results_map = execute_logreg_model(X, y, X_val, y_val, results_map)
        #acc, pre, rec, f1, results_map = execute_svm_model(X, y, X_val, y_val, results_map)
        #acc, pre, rec, f1, results_map = execute_svm_linear_model(X, y, X_val, y_val, results_map)
        acc, pre, rec, f1, results_map = execute_knn_model(X, y, X_val, y_val, results_map)
        acc, pre, rec, f1, results_map = execute_nb_model(X, y, X_val, y_val, results_map)
        acc, pre, rec, f1, results_map = execute_rf_model(X, y, X_val, y_val, results_map)
        results = process_results_map(results_map, "Traditional Classifiers", y_val)


if __name__ == '__main__':
    main()
