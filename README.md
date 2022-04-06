# Wine Quality Classification with Machine Learning

## Overview
* Created a machine learning classification model to to predict wine quality (67% accuracy), based on the wines' chemical characteristics.
* Used statistical analysis (multiple linear OLS regression) to highlight key predictors of quality.
* Built and compared three models: Multinomial Logistic Regression, Decision Tree and Random Forest classifiers.
* Optimized the models with GridSearchCV to tune hyperparameters and reach the best model.

## Code and Resources Used

**Programming Language:** Python  
**Packages:** pandas, numpy, scikit-learn, stats...  
**Dataset Source:** https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

## EDA

![](quality_dist.png)

![](characteristics_heatmap.png)

![](quality_alcohol_reg.png)

## Statistical Analysis

![](multi_regression.PNG)

## Model Building

First I scaled the feature data to improve the training of the models.

Then I tried three different models:
* Multionomial Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier

## Model Performance
