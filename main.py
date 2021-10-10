from data import generate_examples_list
from ml import calculate_statistical_features, separate_training_validation, execute_ml_models


def main():
    #define params
    folder_path = "dataset/"
    atlas_id = 1
    correlation = "Pearson Correlation and Fisher Normalization"
    
    #execute 
    example_list = generate_examples_list(folder_path, atlas_id, correlation)
    exp_list = calculate_statistical_features(example_list)
    X, X_val, y, y_val = separate_training_validation(exp_list)
    execute_ml_models(X, y, X_val, y_val)


if __name__ == '__main__':
    main()
