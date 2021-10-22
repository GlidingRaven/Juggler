import numpy as np
import pandas as pd
from sklearn import model_selection, datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
import pickle

PATH_TO_DATA = 'data/'

class BasePredictor():
    def __init__(self, filename=False):
        if not filename:
            print('Predictor empty. Fit it')
        else:
            full_path = PATH_TO_DATA + filename
            self.pipeline = pickle.load(open(full_path, 'rb'))

    def fit(self, filename):
        pass

    def save(self, filename):
        full_path = PATH_TO_DATA + filename
        pickle.dump(self.pipeline, open(full_path, 'wb'))  # save model

    def predict(self, raw_input):
        pass

# Linear regression with PolynomialFeatures
class ActionPredictor(BasePredictor):
    def fit(self, filename):
        full_path = PATH_TO_DATA + filename
        data = pd.read_csv(full_path)
        data = data.drop(['score'], axis=1).sample(frac=1)
        full_train_data = data
        targets = ['alpha', 'beta', 'z_vel', 'delay']

        train_labels = full_train_data[targets]
        train_data = full_train_data.drop(targets, axis=1)

        self.poly = PolynomialFeatures(2)
        self.scaler = StandardScaler()
        self.regressor = MultiOutputRegressor(linear_model.SGDRegressor(random_state=0))

        self.pipeline = Pipeline(steps=[('polynomial', self.poly), ('scaling', self.scaler), ('regression', self.regressor)])
        self.pipeline.fit(train_data, train_labels)

        pred = self.pipeline.predict(train_data)
        mae = metrics.mean_absolute_error(train_labels, pred, multioutput='raw_values')
        print('Model Fitted')
        print('MAE = ', *mae)

    def predict(self, raw_input):
        data = np.array(raw_input, dtype=np.float64).reshape(1, -1)
        pred = self.pipeline.predict(data)
        return pred[0]


class ZPredictor(BasePredictor):
    def fit(self, filename):
        full_path = PATH_TO_DATA + filename
        data = pd.read_csv(full_path)
        full_train_data = data.iloc[0:36, :]
        full_test_data = data.iloc[36:, :]

        train_labels = full_train_data[['z_real']]
        test_labels = full_test_data[['z_real']]
        train_data = full_train_data[['y', 'y2']]
        test_data = full_test_data[['y', 'y2']]

        self.poly = PolynomialFeatures(2)
        self.scaler = StandardScaler()
        self.regressor = MultiOutputRegressor(
            linear_model.SGDRegressor(random_state=0, max_iter=1000, penalty='l1', alpha=1))

        self.pipeline = Pipeline(steps=[('polynomial', poly), ('scaling', scaler), ('regression', regressor)])
        self.pipeline.fit(train_data, train_labels)


        pred = self.pipeline.predict(test_data)
        mae = metrics.mean_absolute_error(test_labels, pred, multioutput='raw_values')
        print('Model Fitted')
        print('MAE = ', *mae)