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


class Cord_finder(BasePredictor):
    def __init__(self, model_name):
        self.models = []

        with open(model_name, "rb") as file:
            while True:
                try:
                    self.models.append(pickle.load(file))
                except EOFError:
                    break

    def predict_z(self, x, y, x2, y2):
        data_1 = np.array([y, x2], dtype=np.float64).reshape(1, -1)
        data_2 = np.array([x, y2], dtype=np.float64).reshape(1, -1)
        pred_1 = self.models[0].predict(data_1)
        pred_2 = self.models[1].predict(data_2)
        return float(pred_1[0][0] + pred_2[0][0]) / 2

    def predict_x_y(self, x, x2, z_cord):
        data_3 = np.array([x, x2, z_cord], dtype=np.float64).reshape(1, -1)
        pred_3 = self.models[2].predict(data_3)
        return np.multiply([pred_3[0][0], pred_3[0][1], z_cord], 10) # return result in milimeters