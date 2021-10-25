from data import generate_examples_list, load_extra_info
from ml import calculate_statistical_features, calculate_statistical_features_gcn, separate_training_validation, execute_logreg_model, execute_svm_model, generate_stellar_graph, create_graph_classification_model, train_gcn_model, execute_gcn_model


def main():

    #model_type = "SVM"
    #model_type = "LR"
    model_type = "GCN"

    #define params for SVM or LR model
    folder_path = "dataset4/"
    atlas_id = 1
    correlation = "Pearson Correlation and Fisher Normalization"
    thresh = 0.21
    binarize_coef = 0.3
    feature_selection_id = 3

    #define params for GCN model
    epochs = 20  
    folds = 5 
    n_repeats = 3 

    #execute 
    extra_info = load_extra_info()
    example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info)

    if (model_type == "SVM" or model_type == "LR"):
        exp_list = calculate_statistical_features(example_list)
        X, X_val, y, y_val = separate_training_validation(exp_list, feature_selection_id)
        results = execute_logreg_model(X, y, X_val, y_val)
        results = execute_svm_model(X, y, X_val, y_val)
    else:
        exp_list = calculate_statistical_features_gcn(example_list)
        graphs, graphs_labels, generator = generate_stellar_graph(exp_list)
        model, generator = train_gcn_model(folds, n_repeats, graphs_labels, epochs, generator)
        execute_gcn_model(model, generator)


if __name__ == '__main__':
    main()
