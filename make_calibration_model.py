# This is copy of ipynb file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, math, time
import pickle
from sklearn import model_selection, datasets, linear_model, metrics, svm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

data = pd.read_csv('data/' + 'calibration_dots.csv')

full_train_data = data.iloc[0:36, :]
full_test_data = data.iloc[36:, :]

print('Shapes of train:{}   test:{}   and all:{}'.format(full_train_data.shape, full_test_data.shape, data.shape))

features = ['y', 'x2']
train_labels = full_train_data[['z_real']]
test_labels = full_test_data[['z_real']]
train_data = full_train_data[features]
test_data = full_test_data[features]

poly1 = PolynomialFeatures(2)
scaler1 = StandardScaler()
regressor1 = linear_model.LinearRegression()

pipeline1 = Pipeline(steps = [('polynomial', poly1), ('scaling', scaler1), ('regression', regressor1)])
pipeline1.fit(train_data, train_labels)

pred = pipeline1.predict(test_data)
mae = metrics.mean_absolute_error(test_labels, pred)
print('MAE = ',  mae)
print('Coefficients: ', *regressor1.coef_)

features = ['x', 'y2']
train_labels = full_train_data[['z_real']]
test_labels = full_test_data[['z_real']]
train_data = full_train_data[features]
test_data = full_test_data[features]

poly2 = PolynomialFeatures(2)
scaler2 = StandardScaler()
regressor2 = linear_model.LinearRegression()

pipeline2 = Pipeline(steps = [('polynomial', poly2), ('scaling', scaler2), ('regression', regressor2)])
pipeline2.fit(train_data, train_labels)

pred = pipeline2.predict(test_data)
mae = metrics.mean_absolute_error(test_labels, pred)
print('MAE = ',  mae)
print('Coefficients: ', *regressor2.coef_)

features = ['x', 'x2', 'z_real']
train_labels = full_train_data[['x_real', 'y_real']]
test_labels = full_test_data[['x_real', 'y_real']]
train_data = full_train_data[features]
test_data = full_test_data[features]

poly3 = PolynomialFeatures(2)
scaler3 = StandardScaler()
regressor3 = MultiOutputRegressor(linear_model.LinearRegression())

pipeline3 = Pipeline(steps = [('polynomial', poly3), ('scaling', scaler3), ('regression', regressor3)])
pipeline3.fit(train_data, train_labels)

pred = pipeline3.predict(test_data)
mae = metrics.mean_absolute_error(test_labels, pred, multioutput='raw_values')
print('MAE = ',  *mae)

print('\nCoefficients of x_real part: ', regressor3.estimators_[0].coef_, '\n')
print('Coefficients of y_real part: ', regressor3.estimators_[1].coef_, '\n')

models = [pipeline1, pipeline2, pipeline3]


with open('data/calibration_models.sav', "wb") as file:
    for model in models:
         pickle.dump(model, file)

