"""
Library of functions to find customers who are likely to churn.

author: Carlos Escamilla
date: October 26, 2023
"""
import os
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constant import (ARTIFACTS_PATH,
                      QT_QPA_PLATFORM,
                      DATASET_PATH,
                      COLUMNS_TO_REMOVE,
                      RESPONSE_VARIABLE,
                      VARIABLES_TO_MAKE_A_BAR_CHART,
                      VARIABLES_TO_MAKE_A_HISTOGRAM,
                      CATEGORIES_TO_ENCODE,
                      TEST_SIZE,
                      LOG_REG_SOLVER,
                      LOG_REG_MAX_ITER,
                      RANDOM_FOREST_PARAM_GRID,
                      RANDOM_FOREST_GRID_CV,
                      LOG_REG_MODEL_NAME,
                      RANDOM_FOREST_MODEL_NAME)
sns.set()

os.environ['QT_QPA_PLATFORM'] = QT_QPA_PLATFORM


def import_data(path):
    """
    Returns dataframe for the csv found at path.

    Args:
        path (str): Path to the csv.

    Return:
        dataset (pd.DataFrame): pandas dataframe with csv data.
    """
    dataset = pd.read_csv(Path(path))
    dataset.drop(columns=['Unnamed: 0'], inplace=True)
    return dataset


def data_summary(dataset):
    """
    Print summary of dataframe dataset:
        1) Data types for each column.
        2) Dataframe shape.

    Args:
        dataset (pd.DataFrame): Pandas dataframe.

    Return:
        None
    """
    print("Data summary:")
    for i, name in enumerate(dataset.columns):
        print(f"{i}) {name} dtype: {dataset.dtypes[i]}")
    print(f"Data shape: {dataset.shape}")


def bar_chart(dataset, column_name, directory_path):
    """
    Create and save a bar chart from a dataframe column.
    The figure is in png format.

    Args:
        dataset (pd.DataFrame): Pandas dataframe.
        column_name (str): Dataframe column of interest.
        directory_path (str): Folder path to save bar chart.

    Return:
        None
    """
    img_path = Path(directory_path).joinpath(f"{column_name}.png")
    counts = dict(dataset[column_name].value_counts())
    plt.figure(figsize=(20, 10))
    plot = plt.bar(counts.keys(), counts.values())
    total = 0
    for key in counts:
        total += counts[key]
    for bar_plot in plot:
        height = bar_plot.get_height()
        plt.annotate(
            f"{height/total*100:.4f}%",
            (bar_plot.get_x() + bar_plot.get_width() / 2, height + .02),
            ha="center", va="bottom", fontsize=12)
    plt.title(column_name)
    plt.ylabel("Counts")
    plt.savefig(img_path)
    plt.close()


def hist_chart(dataset, column_name, directory_path):
    """
    Create and save a histogram figure from a dataframe column.
    The figure is in png format.

    Args:
        dataset (pd.DataFrame): Pandas dataframe.
        column_name (str): Dataframe column of interest.
        directory_path (str): Folder path to save the histogram.

    Return:
        None
    """
    img_path = Path(directory_path).joinpath(f"{column_name}.png")
    plt.figure(figsize=(20, 10))
    dataset[column_name].hist()
    plt.title(f"Histogram: {column_name}")
    plt.ylabel("Counts")
    plt.savefig(img_path)
    plt.close()


def perform_eda(dataset, output_path: str):
    """
    Perform Exploratory Data Analysis on dataframe dataset,
    and save figures.

    Args:
        dataset (pd.DataFrame): Pandas dataframe.
        output_path (str): Folder path to save artifacts.

    Return:
        None
    """
    data_summary(dataset)
    img_path = Path(output_path).joinpath("images").joinpath("eda")
    img_path.mkdir(parents=True, exist_ok=True)
    for category in VARIABLES_TO_MAKE_A_BAR_CHART:
        bar_chart(dataset, category, img_path)
    for var in VARIABLES_TO_MAKE_A_HISTOGRAM:
        hist_chart(dataset, var, img_path)

    # Correlation plot: Credit limit vs. Open to buy credit line
    plt.figure(figsize=(20, 10))
    cred_limit = dataset['Credit_Limit'].values
    avg_open_to_buy = dataset['Avg_Open_To_Buy'].values
    plt.scatter(cred_limit, avg_open_to_buy, alpha=0.5)
    plt.title('Correlation plot: Credit limit vs. Open to buy credit line')
    plt.xlabel("Credit_Limit")
    plt.ylabel("Avg_Open_To_Buy")
    plt.savefig(img_path.joinpath('corr_credit_limit_vs_open_to_buy.png'))
    plt.close()

    # Distribution: Total transaction plot
    plt.figure(figsize=(20, 10))
    sns.histplot(dataset['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Density: Total transaction count plot')
    plt.savefig(img_path.joinpath('Total_Trans_Ct.png'))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(dataset.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Correlation heatmap")
    plt.savefig(img_path.joinpath('correlation_heatmap.png'))
    plt.close()


def encoder_helper(dataset, category_lst, response='Churn'):
    """
    Helper function to turn each categorical column into a new column with
    propotion of the response variable.

    Args:
        dataset (pd.DataFrame): Pandas dataframe.
        category_lst (list): List of columns that contain categorical features.
        response (str): String of response name. Default value is 'Churn'.

    Return:
        dataset (pd.DataFrame): Pandas dataframe with new columns.
    """
    if response == 'Churn':
        dataset[response] = dataset['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        dataset = dataset.drop(columns=['Attrition_Flag'])
    for category in category_lst:
        lst = []
        groups = dataset.groupby(category).mean()[response]
        for val in dataset[category]:
            lst.append(groups.loc[val])
        dataset[f'{category}_{response}'] = lst
    return dataset


def perform_feature_engineering(
        dataset,
        category_lst,
        feat_to_remove,
        test_size,
        response='Churn'):
    '''
    Applies feature engineering and prepare dataset for training.

    Args:
        dataset (pd.DataFrame): Pandas dataframe.
        category_lst (list): List of columns that contain categorical features.
        feat_to_remove (list): List of columns to remove from dataframe.
        test_size (float): Proportion of the test set from total examples.
        response (str): String of response name. Default value is 'Churn'.

    Return:
        x_train (np.array): Numpy array with training examples.
        x_test (np.array): Numpy array with test examples.
        y_train (np.array): Numpy array with response variable for training set.
        y_test (np.array): Numpy arrray with response variable for test set
    '''
    dataset = encoder_helper(dataset, category_lst, response)
    response_values = dataset[response].values
    dataset = dataset.drop(columns=feat_to_remove + category_lst + [response])
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, response_values, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(train_results,
                                test_results,
                                model_name,
                                output_path):
    '''
    Stores classification report for training and testing results.

    Args:
        train_results (str): Metrics report for training values.
        test_results (str):  Metrics report for test values.
        model_name (str): Name of the model included report.
        output_path (str): Directory to save artifacts.

    Return:
        None
    '''
    plt.rc('figure', figsize=(8, 5))
    plt.text(0.01, 0.99, f"{model_name} Train", {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.59, str(train_results), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, f"{model_name} Test", {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.1, str(test_results), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(output_path.joinpath(f"{model_name}_report.png"))
    plt.close()


def feature_importance_plot(
        model,
        features,
        output_path,
        is_shap_analysis=True):
    '''
    Creates and stores the feature importances in output_path.

    Args:
        model (pkl): pkl object with the model to evaluate.
        features (pd.DataFrame): pandas dataframe of X values (features).
        output_path (str): Path to store model artifacts.
        is_shap_analysis (bool): if True (False), shap (feat importance) analysis is run.
        is_shap_analysis is False, the model should have a feature_importances_
        attribute.

    Return:
        None
    '''
    if is_shap_analysis:
        plt.figure()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        shap.summary_plot(
            shap_values,
            features,
            plot_type="bar",
            show=False,
            plot_size=[
                25,
                15])
        plt.savefig(output_path.joinpath('shap.png'))
        plt.close()
    else:
        importances = model.feature_importances_
        # Sort importances in descending order
        indices = np.argsort(importances)[::-1]
        names = [features.columns[i] for i in indices]
        plt.figure(figsize=(25, 15))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(features.shape[1]), importances[indices])
        plt.xticks(range(features.shape[1]), names, rotation=90)
        plt.savefig(output_path.joinpath('feat_importance.png'))
        plt.close()


def model_report(x_values, target_values, model):
    """
    Creates a classification report with the scores for model predictions.

    Args:
        x_values (np.array):  Numpy array with feature values.
        target_values (np.array): Numpy array with response values
        model (sklearn model): Scikit-learn classification model with predict method.

    Return:
        results (str): classification report.
    """
    pred_values = model.predict(x_values)
    results = classification_report(target_values, pred_values)
    return results


def roc_curve(x_values, target_values, models, report_path):
    """
    Creates ROC curve for a list of clasification models.

    Args:
        x_values (np.array):  Numpy array with feature values.
        target_values (np.array): Numpy array with response values
        models (list): List of sklearn models.
        report_path (str): Directory to save artifact (ROC curve).

    Return:
        None
    """
    plt.figure(figsize=(8, 8))
    fig_axes = plt.gca()
    for model in models:
        plot_roc_curve(model, x_values, target_values, ax=fig_axes, alpha=0.8)
    plt.savefig(report_path.joinpath('ROC_curve.png'))
    plt.close()


def save_model(output_path, model, model_name):
    """
    Saves model in an specify path.

    Args:
        output_path (str): directory to save model.
        model (sklearn model): classification model.
        model_name (str): Name of the model to be saved

    Return:
        None
    """
    model_path = Path(output_path).joinpath("models")
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path.joinpath(model_name))


def train_models(x_train, x_test, y_train, y_test, output_path):
    """
    Train, and store model results: models, images, and scores.

    Args:
        x_train (np.array): Training features.
        x_test (np.array): Test features.
        y_train (np.array): Training response variable.
        y_test (np.array): Test response variable.
        output_path (str): directory to save model artifacts.

    Return:
        None
    """
    lrc = LogisticRegression(solver=LOG_REG_SOLVER, max_iter=LOG_REG_MAX_ITER)
    lrc.fit(x_train, y_train)
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=RANDOM_FOREST_PARAM_GRID,
        cv=RANDOM_FOREST_GRID_CV)
    cv_rfc.fit(x_train, y_train)

    report_path = Path(output_path).joinpath("images").joinpath('results')
    report_path.mkdir(parents=True, exist_ok=True)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_train,
        report_path,
        False)
    feature_importance_plot(cv_rfc.best_estimator_, x_test, report_path, True)

    models = [(cv_rfc.best_estimator_, 'Random_Forest'),
              (lrc, 'Logistic_Regression')]
    for model, name in models:
        train_results = model_report(x_train, y_train, model)
        test_results = model_report(x_test, y_test, model)
        classification_report_image(train_results,
                                    test_results,
                                    name,
                                    report_path
                                    )

    roc_curve(x_test, y_test, [cv_rfc.best_estimator_, lrc], report_path)

    save_model(output_path, cv_rfc.best_estimator_, RANDOM_FOREST_MODEL_NAME)
    save_model(output_path, lrc, LOG_REG_MODEL_NAME)


if __name__ == '__main__':
    df = import_data(DATASET_PATH)
    perform_eda(df, ARTIFACTS_PATH)
    train_features, test_features, train_response, test_response = perform_feature_engineering(
        df, CATEGORIES_TO_ENCODE, COLUMNS_TO_REMOVE, TEST_SIZE, RESPONSE_VARIABLE)
    train_models(
        train_features,
        test_features,
        train_response,
        test_response,
        ARTIFACTS_PATH)
