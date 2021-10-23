import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random, math, time, sys, pickle
from sklearn import model_selection, datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from Juggler import Plotter, Ball_detection, Configurator
import warnings
warnings.filterwarnings('ignore')

file_name_1 = 'cam/3a.avi'
file_name_2 = 'cam/3b.avi'
color_ranges_1 = (12,83,194),(35,193,255)
color_ranges_2 = (10,140,143),(26,255,255)
cap1 = cv2.VideoCapture(file_name_1)
cap2 = cv2.VideoCapture(file_name_2)
model_1 = pickle.load(open('data/calibration_model_1.sav', 'rb'))
model_2 = pickle.load(open('data/calibration_model_2.sav', 'rb'))

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
app = QApplication(sys.argv)
QApplication.setStyle(QStyleFactory.create('Plastique'))
myGUI = Plotter.CustomMainWindow(queue_size = 200, y_limit = 400, start_offset = 200, ylabel='radius')
mySrc = Plotter.Communicate()
mySrc.data_signal.connect(myGUI.addData_callbackFunc)

class Avg_value(): # Moving average value
    def __init__(self, size = 30):
        self.array = [0]
        self.size = size

    def add(self, val):
        if len(self.array) >= self.size:
            self.size.pop(0)
        self.array.append(val)

    def get(self):
        return sum(self.array) / len(self.array)

x1, x2, y1, y2, r1, r2 = Avg_value(), Avg_value(), Avg_value(), Avg_value(), Avg_value(), Avg_value()
location_screen = Plotter.Location_screen()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        mid1 = Ball_detection.prepare_frame(frame1, color_ranges_1)
        mid2 = Ball_detection.prepare_frame(frame2, color_ranges_2)
        circles1 = Ball_detection.find_circle( mid1 )
        circles2 = Ball_detection.find_circle( mid2 )
        Ball_detection.draw_circles(frame1, circles1)
        Ball_detection.draw_circles(frame2, circles2)
        cv2.imshow('1', np.hstack((frame1, frame2)))
        # cv2.imshow('2', np.hstack((mid1, mid2)))
        if circles1 is not None and circles2 is not None:
            # print([circles1[0][0][1], circles2[0][0][1]])
            data_1 = np.array([circles1[0][0][1], circles2[0][0][1]], dtype=np.float64).reshape(1, -1)
            pred_1 = model_1.predict(data_1)
            z_cord = pred_1[0][0]

            data_2 = np.array([circles1[0][0][0], circles1[0][0][1], z_cord], dtype=np.float64).reshape(1, -1)
            pred_2 = model_2.predict(data_2)
            cord = np.multiply([pred_2[0][0], pred_2[0][1], z_cord], 10)
            # print(cord)
            mySrc.data_signal.emit(cord[2]+0)
            for_show = location_screen.make_screen(cord, resize=False)
            cv2.imshow('location', for_show)

    else:
        cap1 = cv2.VideoCapture(file_name_1)
        cap2 = cv2.VideoCapture(file_name_2)
        # print('Error on read(): ', ret1, ret2)
        # cap1.release()
        # cap2.release()
        # break


    # for_show = location_screen.make_screen(cord, resize=True)
    # cv2.imshow('location', for_show)

    k = cv2.waitKey(Configurator.frame_period) & 0xFF
    if k == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
# sys.exit(app.exec_())