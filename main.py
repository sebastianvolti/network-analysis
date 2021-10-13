from data import generate_examples_list
from ml import calculate_statistical_features, separate_training_validation, execute_logreg_model, execute_svm_model


def main():

    #define params
    folder_path = "dataset1/"
    atlas_id = 1
    correlation = "Pearson Correlation and Fisher Normalization"
    thresh = 0.21
    binarize_coef = 0.3


    #execute 
    example_list = generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef)
    exp_list = calculate_statistical_features(example_list)
    X, X_val, y, y_val = separate_training_validation(exp_list)
    results = execute_logreg_model(X, y, X_val, y_val)
    results = execute_svm_model(X, y, X_val, y_val)


if __name__ == '__main__':
    main()
