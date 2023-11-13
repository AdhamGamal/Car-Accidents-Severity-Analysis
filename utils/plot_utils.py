import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


import plotly
plotly.offline.init_notebook_mode()


# ****************************************************************************
# ****************************************************************************
def display_value_counts_with_order(df, columns, cols_per_row, fig=(10,20)):
    """
    Displays count plots of categorical columns with a specified order of values.

    Parameters:
    - df: Pandas DataFrame.
    - columns (list): List of column names to display.
    - cols_per_row (int): Number of columns to display per row.
    - fig (tuple): Figure size.

    Returns:
    None
    """
    total_size = len(columns)
    nrows = int(np.ceil(len(columns) / cols_per_row))

    fig, axes = plt.subplots(nrows=nrows, ncols=cols_per_row, figsize=fig)
    axes = axes.flatten()
    for ax, column in zip(axes[:total_size], columns[:total_size]):
        top_values = df[column].value_counts().sort_values(ascending=False)
        order = top_values.index
        sns.countplot(x=column, data=df[df[column].isin(top_values.index)], order=order, palette='viridis', ax=ax)

        ax.set_title(column)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('')

    plt.tight_layout()
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_value_counts(df, columns, cols_per_row, fig=(10,20)):
    """
    Displays count plots of categorical columns.

    Parameters:
    - df: Pandas DataFrame.
    - columns (list): List of column names to display.
    - cols_per_row (int): Number of columns to display per row.
    - fig (tuple): Figure size.

    Returns:
    None
    """
    total_size = len(columns)
    nrows = int(np.ceil(len(columns) / cols_per_row))

    fig, axes = plt.subplots(nrows=nrows, ncols=cols_per_row, figsize=fig)
    axes = axes.flatten()
    for ax, column in zip(axes[:total_size], columns[:total_size]):
        sns.countplot(x=column, data=df, palette='viridis', ax=ax)

        ax.set_title(column)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('')

    plt.tight_layout()
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_distributions(df, columns, cols_per_row, fig=(10,20)):
    """
    Displays histograms of numerical columns.

    Parameters:
    - df: Pandas DataFrame.
    - columns (list): List of column names to display.
    - cols_per_row (int): Number of columns to display per row.
    - fig (tuple): Figure size.

    Returns:
    None
    """
    total_size = len(columns)
    nrows = int(np.ceil(len(columns) / cols_per_row))

    fig, axes = plt.subplots(nrows=nrows, ncols=cols_per_row, figsize=fig)
    axes = axes.flatten()
    for ax, column in zip(axes[:total_size], columns[:total_size]):
        sns.histplot(df[column], bins=20, kde=True, color='skyblue', ax=ax)

        ax.set_title(f'Distribution of {column}')
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('')

    plt.tight_layout()
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_feature_distribution(df, column):
    """
    Displays the distribution of values for a specific column.

    Parameters:
    - df: Pandas DataFrame.
    - column (str): Name of the column to display.

    Returns:
    None
    """
    plt.figure(figsize=(20, 6))
    order = df[column].value_counts().index
    sns.countplot(x=column, data=df, order=order, palette='viridis')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=90)
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_features_relationship(df, feature1, feature2, rotate_x_axis=False):
    """
    Displays the relationship between two categorical features using a count plot.

    Parameters:
    - df: Pandas DataFrame.
    - feature1 (str): Name of the first feature.
    - feature2 (str): Name of the second feature.
    - rotate_x_axis (bool): Whether to rotate the x-axis labels.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature1, hue=feature2, data=df, palette='viridis')
    plt.title(f'Distribution of {feature2} over {feature1}')
    if rotate_x_axis:
        plt.xticks(rotation=90)
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_correlation(df, method):
    """
    Displays a heatmap of the correlation matrix of the DataFrame.

    Parameters:
    - df: Pandas DataFrame.
    - method (str): Correlation method ('pearson', 'kendall', 'spearman').

    Returns:
    None
    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(method), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show();


# ****************************************************************************
# ****************************************************************************
def plot_oversampling_comparison(results_data, metrics):
    """
    Plots a comparison of oversampling techniques for different models.

    Parameters:
    - results_data (dict): Dictionary containing oversampling techniques, models, and their results.
    - metrics (list): List of metrics to compare.

    Returns:
    None
    """
    oversample_techniques = list(results_data.keys())
    models = list(results_data[oversample_techniques[0]].keys())

    fig = go.Figure()

    for model, metric in product(models, metrics):
        metric_results = [results_data[technique][model][metric] for technique in oversample_techniques]
        fig.add_trace(go.Scatter(x=oversample_techniques, y=metric_results, mode='markers+lines', name=f'{model} {metric}'))

    fig.update_layout(
        xaxis=dict(title='Model'),
        yaxis=dict(title='Metric'),
        title=f'Different Over Sampling Techniques over Different Models',
        legend=dict(title='Model'),
    )

    fig.show();


# ****************************************************************************
# ****************************************************************************
def plot_models_comparison(results_data, metrics):
    """
    Plots a comparison of test and train accuracy for different models.

    Parameters:
    - results_data (dict): Dictionary containing models and their results.
    - metrics (list): List of metrics to compare.

    Returns:
    None
    """
    models = list(results_data.keys())
    values = results_data.values()

    fig = go.Figure()

    for metric in metrics:
        metric_results = [result[metric] for result in values]
        fig.add_trace(go.Scatter(x=models, y=metric_results, mode='markers+lines', name=metric))

    fig.update_layout(
        xaxis=dict(title='Model'),
        yaxis=dict(title='Metric'),
        title='Metrics for Different Models',
        legend=dict(title='Accuracy Type'),
    )

    fig.show();


# ****************************************************************************
# ****************************************************************************
def plotCrossValidationComparison(model_name, model_scores):
    """
    Plot a comparison of cross-validation scores for multiple models.

    Parameters:
    - model_name (str): The name or identifier of the model.
    - model_scores (array): Array of model scores obtained from cross-validation.

    Returns:
    None
    """
    indics = range(1, model_scores.shape[1] + 1)
    models = [f'Model {index}' for index in indics]
    ranks = [f'Rank {index}' for index in indics]

    for scores in model_scores:
        plt.plot(ranks, scores, marker='o')

    plt.title(f'{model_name} Cross Validation Comparison')
    plt.xlabel("CV test fold")
    plt.ylabel("Metric")
    plt.legend(labels=models, fontsize="large")
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_confusion_matrix(model_name, best_model, X_test, y_test):
    """
    Display the confusion matrix for a given model on the test set.

    Parameters:
    - model_name (str): The name or identifier of the model.
    - best_model: The best-performing model obtained from hyperparameter tuning.
    - X_test: Testing features.
    - y_test: Testing target variable.

    Returns:
    None
    """
    print("Confusion Matrix:")
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=['Fatal', 'Serious', 'Slight'], normalize='true')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show();
