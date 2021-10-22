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

train_labels = full_train_data[['z_real']]
test_labels = full_test_data[['z_real']]
train_data = full_train_data[['y', 'y2']]
test_data = full_test_data[['y', 'y2']]


poly = PolynomialFeatures(2)
scaler = StandardScaler()
regressor = MultiOutputRegressor(linear_model.SGDRegressor(random_state = 0, max_iter=1000, penalty='l1', alpha=1))

pipeline = Pipeline(steps = [('polynomial', poly), ('scaling', scaler), ('regression', regressor)])
pipeline.fit(train_data, train_labels)

pickle.dump(pipeline, open('data/calibration_model_11.sav', 'wb')) # save model

pred = pipeline.predict(test_data)
mae = metrics.mean_absolute_error(test_labels, pred, multioutput='raw_values')
print('MAE = ',  *mae)
# mse = metrics.mean_squared_error(test_labels, pred, multioutput='raw_values')
# print('standart error = ',  math.sqrt(*mse))
print('Coefficients: ', regressor.estimators_[0].coef_)


train_labels = full_train_data[['x_real', 'y_real']]
train_data = full_train_data[['x', 'y', 'z_real']]
test_labels = full_test_data[['x_real', 'y_real']]
test_data = full_test_data[['x', 'y', 'z_real']]

# train_data = np.hstack((train_data, train_data**2))
# test_data = np.hstack((test_data, test_data**2))
# test_labels, test_data = train_labels, train_data

poly = PolynomialFeatures(2)
scaler = StandardScaler()
regressor = MultiOutputRegressor(linear_model.SGDRegressor(random_state = 0))

pipeline = Pipeline(steps = [('polynomial', poly), ('scaling', scaler), ('regression', regressor)])
pipeline.fit(train_data, train_labels)

pickle.dump(pipeline, open('data/calibration_model_22.sav', 'wb')) # save model

pred = pipeline.predict(train_data)
mae = metrics.mean_absolute_error(train_labels, pred, multioutput='raw_values')
print('MAE = ',  *mae)

pred = pipeline.predict(test_data)
mae = metrics.mean_absolute_error(test_labels, pred, multioutput='raw_values')
mse = metrics.mean_squared_error(test_labels, pred, multioutput='raw_values')
print('MAE = ',  *mae)
# print('standart error = ',  *np.sqrt(mse))

print('\nCoefficients of x_real part: ', regressor.estimators_[0].coef_, '\n')
print('Coefficients of y_real part: ', regressor.estimators_[1].coef_, '\n')