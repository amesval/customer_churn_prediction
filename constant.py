"""
Configuration parameters to setup pipeline of Customer churn prediction.

author: Carlos Escamilla
date: October 26, 2023
"""
ARTIFACTS_PATH = './'  # root directory
QT_QPA_PLATFORM = 'offscreen'
DATASET_PATH = './data/bank_data.csv'
COLUMNS_TO_REMOVE = ['CLIENTNUM']
RESPONSE_VARIABLE = 'Churn'
VARIABLES_TO_MAKE_A_BAR_CHART = [
    'Attrition_Flag',
    'Gender',
    'Dependent_count',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
VARIABLES_TO_MAKE_A_HISTOGRAM = [
    'Customer_Age',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Avg_Utilization_Ratio']
CATEGORIES_TO_ENCODE = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
TEST_SIZE = 0.3
LOG_REG_SOLVER = 'lbfgs'
LOG_REG_MAX_ITER = 3000
RANDOM_FOREST_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']}
RANDOM_FOREST_GRID_CV = 5
LOG_REG_MODEL_NAME = "logistic_reg_model.pkl"
RANDOM_FOREST_MODEL_NAME = "random_forest_model.pkl"
