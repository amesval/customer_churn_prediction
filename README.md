# Predict Customer Churn

- Customer Churn prediction is an important factor for business success, and is the focus of this project. The current library covers different steps to succeed on this task, including: pre-process data, training a ML model, predict labels for unknown data, evaluation, and model interpretability.

- This project is part of the ML DevOps Engineer Nanodegree Udacity. 

## Project Description
In this work it is used a credit card customer dataset from Kaggle (https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). The dataset consist of 10000 customers with a set of 21 demographic features, such as age, salary, marital_status, gender, credit limit, etc. This dataset is highly imbalanced, with only 16% of customer churn.

The next table provides a description of the customer variables:

| Feature | Description |
| ------- | ----------- |
| CLIENTNUM | Unique identifier for the customer holding the account |
| Attrition_Flag | If the account is closed then 1 else 0 |
| Customer_Age | Customer's Age in Years |
| Gender | M=Male, F=Female |
| Dependent_count | Number of dependents |
| Education_Level | Educational qualification of the account holder |
| Marital_Status | Married, Single, Divorced, Unknown |
| Income_Category | Annual income category of the account holder |
| Card_Category | Type of Card (Blue, Silver, Gold, Platinum) |
| Months_on_book | Period of relationship with bank |
| Total_Relationship_Count | Total number of products held by the customer |
| Months_Inactive_12_mon | Number of months inactive in the last 12 months |
| Contacts_Count_12_mon | Number of contacts in the last 12 months |
| Credit_Limit | Credit limit on the credit card |
| Total_Revolving_Bal | Total revolving balance on the credit card |
| Avg_Open_To_Buy | Open to buy credit line (Average of last 12 months) |
| Total_Amt_Chng_Q4_Q1 | Change in transaction amount (Q4 over Q1) |
| Total_Trans_Amt | Total transaction amount (Last 12 months) |
| Total_Trans_Ct | Total transaction count (Last 12 months) |
| Total_Ct_Chng_Q4_Q1 | Change in transaction count (Q4 over Q1) |
| Avg_Utilization_Ratio | Average card utilization ratio |

The Attrition_Flag tell us if a customer churns or not. In other words, this is the response variable to predict.

The library includes the following steps:  
&ensp; &ensp; a) Import data  
&ensp; &ensp; b) Exploratory data analysis  
&ensp; &ensp; c) Feature engineering  
&ensp; &ensp; d) Training models (Random forest, and Logistic regression)  
&ensp; &ensp; e) Evaluation report

## Files and data description
The project have the next tree structure:

root/  
&ensp; \- churn_library.py  
&ensp; \- churn_script_logging_and_tests.py  
&ensp; \- constant.py  
data/  
&ensp; \- bank_data.csv

| File | Description |
| ---- | ----------- |
| churn_library.py | Main file to run the ML pipeline |
| churn_script_logging_and_tests.py| Perform a test run and log process for inspection |
| constant.py | Configuration parameters and hyperparameters |
| bank_data.csv | Customer churn dataset |

After running **churn_library.py**, new directories are created to save artifacts:

root/  
&ensp; images/  
&ensp; &ensp; eda/  
&ensp; &ensp; &ensp; \- <ARTIFACT_NAME>.png   
&ensp; &ensp; results/  
&ensp; &ensp; &ensp; \- <ARTIFACT_NAME>.png  
&ensp; models/  
&ensp; &ensp; &ensp; \- <MODEL_NAME>.pkl

To inspect the process, you can make a test run (optional) with **churn_script_logging_and_tests.py**, and a log file is created:

root/  
&ensp; logs/  
&ensp; &ensp; \- churn_library.log
## Setup

1. Create a conda environment:
```
conda create --name <ENV_NAME> python==3.6
```
2. Change to conda environment:
```
conda activate <ENV_NAME>
```
3. Move to root folder of this project.

4. Install requirements:
```
pip install -r requirements.txt
```

## Running Files

1. Move to root folder.

2. Run code from the terminal:
```
python churn_library.py
```
As explained in the previous section (*Files and data description*), new files are created after running step 2.  
Note: The config and training parameters can be modify directly in **constant.py**

Optional: If you want to inspect and log the process, enter:
```
python churn_script_logging_and_tests.py
```
The logs can be reviewed in **logs/churn_library.log** file.  



