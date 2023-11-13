from utils import plot_utils
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

pd.set_option("display.max_rows", None)


# ****************************************************************************
# ****************************************************************************
def pipeline(X, y, with_over_sampling=False):
    """
    Build and evaluate machine learning models using a pipeline.

    Parameters:
    - X: Features.
    - y: Target variable.
    - with_over_sampling (bool): Whether to include oversampling techniques.

    Returns:
    dict: Dictionary containing model names, best hyperparameters, train accuracy, and test accuracy.
    """
    state = 42

    # models = [
    #     ("Random Forest", RandomForestClassifier(random_state=state), {'model__n_estimators': [10 , 20, 40, 80], 'model__max_depth': [5, 10, 15]}),
    #     ("Gradient Boosting", GradientBoostingClassifier(random_state=state), {'model__n_estimators': [10 , 20, 40, 80], 'model__max_depth': [5, 10, 15]}),
    #     ("XGBoost", XGBClassifier(random_state=state), {'model__n_estimators': [10 , 20, 40, 80], 'model__max_depth': [5, 10, 15]}),
    #     ("SVM", SVC(random_state=state), {'model__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
    #     ("Naive Bayes", MultinomialNB(), {'model__alpha': [0.001, 0.01, 0.1, 0.5, 1.0]}),
    # ]

    models = [
        ("Random Forest", RandomForestClassifier(random_state=state), {'model__n_estimators': [50, 100, 200, 400]}),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=state), {'model__n_estimators': [50, 100, 200, 400]}),
        ("XGBoost", XGBClassifier(random_state=state), {'model__n_estimators': [50, 100, 200, 400]}),
        ("SVM", SVC(random_state=state), {'model__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
        ("Naive Bayes", MultinomialNB(), {'model__alpha': [0.001, 0.01, 0.1, 0.5, 1.0]}),
    ]

    over_sampling_techniques = [ADASYN(random_state=state), SMOTE(random_state=state), BorderlineSMOTE(random_state=state)]

    if with_over_sampling:
        oversampling_data = {}
        for technique in over_sampling_techniques:
            print("*******************************************************************************")
            print("*******************************************************************************")
            print(f"Using {technique.__class__.__name__} Over Sampling...")
            print("*******************************************************************************")
            technique_name, results = model_pipeline(models, X, y, technique)
            oversampling_data[technique_name] = results
        return oversampling_data
    else:
        return model_pipeline(models, X, y)


# ****************************************************************************
# ****************************************************************************
def model_pipeline(models, X, y, over_sampling_technique=None):
    """
    Build and evaluate machine learning models using a pipeline.

    Parameters:
    - models: List of tuples containing model names, model instances, and hyperparameter grids.
    - X: Features.
    - y: Target variable.
    - over_sampling_technique: Oversampling technique (default is None).

    Returns:
    dict: Dictionary containing model names, best hyperparameters, train accuracy, and test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=14)

    results_data = {}

    for model_name, model, param_grid in models:

        if over_sampling_technique:
            pipeline = Pipeline([
                ('smote', over_sampling_technique),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])

        print(f"Running GridSearchCV for {model_name}...")

        results_data[model_name] = gridSearch(pipeline, param_grid, X_train, X_test, y_train, y_test)

    if over_sampling_technique:
        return over_sampling_technique.__class__.__name__, results_data
    else:
        return results_data


# ****************************************************************************
# ****************************************************************************
def gridSearch(model, parameters, X_train, X_test, y_train, y_test):
    """
    Perform grid search for hyperparameter tuning and evaluate the model.

    Parameters:
    - model: The machine learning model to be tuned and evaluated.
    - parameters: The hyperparameter grid for grid search.
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training target variable.
    - y_test: Testing target variable.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    model_name = best_model.__class__.__name__
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    # Calculate and print additional metrics
    y_pred = best_model.predict(X_test)
    
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    test_accuracy = accuracy_score(y_test, y_pred)
    class_repo = classification_report(y_test, y_pred, target_names=['Fatal', 'Serious', 'Slight'])

    print(f"Best Hyperparameters for {model_name}: {best_parameters}")
    print(f"Best Accuracy: {best_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")
    print('Classification Report')
    print(class_repo, '\n')
    plot_utils.display_confusion_matrix(model_name, best_model, X_test, y_test)

    results = pd.DataFrame(grid_search.cv_results_).sort_values(by=['rank_test_score']).iloc[0:5]
    model_scores = results.filter(regex=r"split\d*_test_score")
    plot_utils.plotCrossValidationComparison(model_name, model_scores.values)


    return {
        'Train Accuracy': best_score,
        'Test Accuracy': test_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
