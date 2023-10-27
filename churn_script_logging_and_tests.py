"""
Module to test churn_library, and log the process.

author: Carlos Escamilla
date: October 26, 2023
"""
import logging
from pathlib import Path
import pytest
from churn_library import (import_data,
                           perform_eda,
                           encoder_helper,
                           perform_feature_engineering,
                           train_models)
from constant import (ARTIFACTS_PATH,
                      DATASET_PATH,
                      VARIABLES_TO_MAKE_A_BAR_CHART,
                      VARIABLES_TO_MAKE_A_HISTOGRAM,
                      CATEGORIES_TO_ENCODE,
                      RESPONSE_VARIABLE,
                      COLUMNS_TO_REMOVE,
                      TEST_SIZE,
                      RANDOM_FOREST_MODEL_NAME,
                      LOG_REG_MODEL_NAME)

log_directory = Path(ARTIFACTS_PATH).joinpath("logs")
log_directory.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=f'{ARTIFACTS_PATH}/logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import_data(import_data_function):
    """
    Test import_data function.
    """
    try:
        dataset = import_data_function(DATASET_PATH)
        pytest.dataset = dataset
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataset.shape[0] > 0
        assert dataset.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_perform_eda(perform_eda_function):
    """
    Test perform_eda function.
    """
    try:
        perform_eda_function(pytest.dataset, ARTIFACTS_PATH)
        directory = Path(ARTIFACTS_PATH).joinpath("images").joinpath("eda")
        columns = VARIABLES_TO_MAKE_A_BAR_CHART + VARIABLES_TO_MAKE_A_HISTOGRAM
        for var in columns:
            filename = directory.joinpath(var + ".png")
            assert filename.exists()
        names = [
            'corr_credit_limit_vs_open_to_buy.png',
            'Total_Trans_Ct.png',
            'correlation_heatmap.png']
        for var in names:
            filename = directory.joinpath(var)
            assert filename.exists()
        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        message = f"Testing perform_eda: The file wasn't found ({filename})"
        logging.error(message)
        raise err


def test_encoder_helper(encoder_helper_function):
    """
    Test encoder_helper function.
    """
    try:
        dataset = encoder_helper_function(pytest.dataset, CATEGORIES_TO_ENCODE, RESPONSE_VARIABLE)
        new_cols = [
            f"{cat}_{RESPONSE_VARIABLE}" for cat in CATEGORIES_TO_ENCODE]
        assert set(new_cols).issubset(set(dataset.columns))
        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        message = f"Testing encoder_helper: KeyError doesn't exist ({err})"
        logging.error(message)
        raise err

    except AssertionError as err:
        message = f"Testing encoder_helper: New columns doesn't \
            have the appropiate name (<NAME>_{RESPONSE_VARIABLE})"
        logging.error(message)
        raise err

    try:
        assert RESPONSE_VARIABLE in dataset.columns
    except AssertionError as err:
        message = f"Testing encoder_helper: Response variable was not created ({RESPONSE_VARIABLE})"
        logging.error(message)
        raise err


def test_perform_feature_engineering(perform_feature_eng_function):
    """
    Test perform_feature_engineering.
    """
    try:
        assert set(COLUMNS_TO_REMOVE).issubset(set(pytest.dataset.columns))

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The columns to be removed are not in the dataset")
        raise err

    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        message = "Testing perform_feature_engineering: test size should be in the (0, 1) range."
        logging.error(message)
        raise ValueError("test size should be in the (0, 1) range.")

    try:
        x_train, x_test, y_train, y_test = perform_feature_eng_function(
            pytest.dataset, CATEGORIES_TO_ENCODE, COLUMNS_TO_REMOVE, TEST_SIZE, RESPONSE_VARIABLE)
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        pytest.x_train = x_train
        pytest.x_test = x_test
        pytest.y_train = y_train
        pytest.y_test = y_test
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except AssertionError as err:
        message = """Testing perform_feature_engineering: \
            Incorrect split produces datasets with 0 examples"""
        logging.error(message)
        raise err


def test_train_models(train_models_function):
    """
    Test train_models.
    """
    try:
        train_models_function(
            pytest.x_train,
            pytest.x_test,
            pytest.y_train,
            pytest.y_test,
            ARTIFACTS_PATH)
        results_dir = Path(ARTIFACTS_PATH).joinpath(
            "images").joinpath("results")
        files = [
            'shap.png',
            'feat_importance.png',
            'Random_Forest_report.png',
            'Logistic_Regression_report.png',
            'ROC_curve.png']
        for file in files:
            file = results_dir.joinpath(file)
            assert file.exists()

    except AssertionError as err:
        message = f"Testing train_models: The file wasn't found ({file})"
        logging.error(message)
        raise err

    try:
        model_dir = Path(ARTIFACTS_PATH).joinpath("models")
        models = [RANDOM_FOREST_MODEL_NAME, LOG_REG_MODEL_NAME]
        for model_name in models:
            model_name = model_dir.joinpath(model_name)
            assert model_name.exists()
        logging.info("Testing train_models: SUCCESS")

    except AssertionError as err:
        message = f"Testing train_models: The model wasn't found ({model_name})"
        logging.error(message)
        raise err


if __name__ == "__main__":
    test_import_data(import_data)
    test_perform_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
