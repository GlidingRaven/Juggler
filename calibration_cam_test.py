import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random, math, time, sys, pickle
from sklearn import model_selection, datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from Juggler import Plotter, Configurator
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

# all cv2 transormations
def batch_prep(frame, blur_n, ranges, kernel_o, kernel_c, kwargs):
    res = cv2.GaussianBlur(frame, tuple(blur_n), 0)  # blure
    res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)  # to HSV
    res = cv2.inRange(res, ranges[0], ranges[1])  # Color filter
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel_o)  # Open filter
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel_c)  # Close filter
    circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT, 2, **kwargs)
    return res, circles

def draw_all_circles(frame, circles):
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        res1, circles1 = batch_prep(frame1, Configurator.blur_n, color_ranges_1, Configurator.kernel_o, Configurator.kernel_c, Configurator.kwargs)
        res2, circles2 = batch_prep(frame2, Configurator.blur_n, color_ranges_2, Configurator.kernel_o, Configurator.kernel_c, Configurator.kwargs)
        draw_all_circles(frame1, circles1)
        draw_all_circles(frame2, circles2)
        cv2.imshow('1', np.hstack((frame1, frame2)))
        cv2.imshow('2', np.hstack((res1, res2)))
        if circles1 is not None and circles2 is not None:
            # print([circles1[0][0][1], circles2[0][0][1]])
            data_1 = np.array([circles1[0][0][1], circles2[0][0][1]], dtype=np.float64).reshape(1, -1)
            pred_1 = model_1.predict(data_1)
            z_cord = pred_1[0][0]

            data_2 = np.array([circles1[0][0][0], circles1[0][0][1], z_cord], dtype=np.float64).reshape(1, -1)
            pred_2 = model_2.predict(data_2)
            cord = np.multiply([pred_2[0][0], pred_2[0][1], z_cord], 10)
            # print(cord)
            mySrc.data_signal.emit(cord[0]+150)
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