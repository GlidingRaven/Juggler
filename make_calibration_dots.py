import cv2
import numpy as np
import pandas as pd
import time, sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Plotter

file_name = 'cam/ball3_zoom.avi'
file_name2 = 'cam/ball3_regular.avi'
options_window = 'Options'
before_window = 'Before'
after_window = 'After'
original_window = 'Original and result'
color_ranges = (12, 158, 176), (27, 255, 255)  # ((0,74,163),(82,255,255)) #((12,156,176),(24,255,255))

cap = cv2.VideoCapture(file_name)
cap2 = cv2.VideoCapture(file_name2)
frame_period = 30
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
kwargs = {'minDist': 200,
          'param1': 80,
          'param2': 80,
          'minRadius': 10,
          'maxRadius': 150}
blur_n = [3, 3]
kernel_o = np.ones((3, 3), np.uint8)
kernel_c = np.ones((9, 9), np.uint8)
alpha = 0.5  # for filter


def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value


def ChangeBlur(value):
    global blur_n
    if value % 2 == 1: blur_n = [value, value]


def ChangeKernelOpen(value):
    global kernel_o
    if value % 2 == 1: kernel_o = np.ones((value, value), np.uint8)


def ChangeKernelClose(value):
    global kernel_c
    if value % 2 == 1: kernel_c = np.ones((value, value), np.uint8)


def ChangeDist(value): kwargs['minDist'] = value if value != 0 else 1


def Change1(value): kwargs['param1'] = value if value != 0 else 1


def Change2(value): kwargs['param2'] = value if value != 0 else 1


def ChangeMinRadius(value): kwargs['minRadius'] = value if value != 0 else 1


def ChangeMaxRadius(value): kwargs['maxRadius'] = value if value != 0 else 1


def ChangeAlpha(value):
    global alpha
    alpha = value / 10


cv2.namedWindow(options_window)
cv2.resizeWindow(options_window, 700, 400);
cv2.createTrackbar('FPS', options_window, 30, 300, ChangeFPS)
cv2.createTrackbar('Blur', options_window, 3, 31, ChangeBlur)
cv2.createTrackbar('Open kernel', options_window, 3, 31, ChangeKernelOpen)
cv2.createTrackbar('Close kernel', options_window, 9, 31, ChangeKernelClose)
cv2.createTrackbar('Hough minDist', options_window, 200, 1000, ChangeDist)
cv2.createTrackbar('Hough param1', options_window, 80, 100, Change1)
cv2.createTrackbar('Hough param2', options_window, 24, 200, Change2)
cv2.createTrackbar('Hough minRadius', options_window, 10, 100, ChangeMinRadius)
cv2.createTrackbar('Hough maxRadius', options_window, 150, 200, ChangeMaxRadius)
cv2.createTrackbar('Alpha (filter)', options_window, 5, 10, ChangeAlpha)

app = QApplication(sys.argv)
QApplication.setStyle(QStyleFactory.create('Plastique'))

myGUI = Plotter.CustomMainWindow(title = "Plotter 1", queue_size=200, y_limit=150, start_offset=75, ylabel='radius')
mySrc = Plotter.Communicate()
mySrc.data_signal.connect(myGUI.addData_callbackFunc)
myGUI2 = Plotter.CustomMainWindow(title = "Plotter 2", queue_size=200, y_limit=150, start_offset=75, ylabel='radius')
mySrc2 = Plotter.Communicate()
mySrc2.data_signal.connect(myGUI2.addData_callbackFunc)

radius = 0
radius2 = 0
last_state = []
last_state2 = []
data = []

to_make = []
for z in range(0,31,100):
    for y in range(-15,16,150):
        for x in range(-15,16,15):
            to_make.append((x,y,z))

while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if ret and ret2:
        blur = cv2.GaussianBlur(frame, tuple(blur_n), 0)  # blure
        blur2 = cv2.GaussianBlur(frame2, tuple(blur_n), 0)
        output = blur.copy()
        output2 = blur2.copy()
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # to HSV
        hsv2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
        ready = cv2.inRange(hsv, color_ranges[0], color_ranges[1])  # Color filter
        ready2 = cv2.inRange(hsv2, color_ranges[0], color_ranges[1])
        ready = cv2.morphologyEx(ready, cv2.MORPH_OPEN, kernel_o)  # Open filter
        ready2 = cv2.morphologyEx(ready2, cv2.MORPH_OPEN, kernel_o)
        ready = cv2.morphologyEx(ready, cv2.MORPH_CLOSE, kernel_c)  # Close filter
        ready2 = cv2.morphologyEx(ready2, cv2.MORPH_CLOSE, kernel_c)

        circles = cv2.HoughCircles(ready, cv2.HOUGH_GRADIENT, 2, **kwargs)
        circles2 = cv2.HoughCircles(ready2, cv2.HOUGH_GRADIENT, 2, **kwargs)

        if to_make:
            output = cv2.putText(output, str(to_make[0]), (50, 50), font, 1, color, 1)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            radius = circles[0][2] * (1 - alpha) + radius * alpha  # exponential moving average filter
            mySrc.data_signal.emit(radius)
            last_state = circles[0]
        else:
            output = cv2.putText(output, '.', (50, 50), font, 1, color, 1)

        if circles2 is not None:
            circles2 = np.round(circles2[0, :]).astype("int")
            for (x, y, r) in circles2:
                cv2.circle(output2, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            radius2 = circles2[0][2] * (1 - alpha) + radius2 * alpha  # exponential moving average filter
            mySrc2.data_signal.emit(radius2)
            last_state2 = circles2[0]
        else:
            mySrc2.data_signal.emit(0)

        cv2.imshow('1', output)
        cv2.imshow('2', output2)
    else:
        cap = cv2.VideoCapture(file_name)
        cap2 = cv2.VideoCapture(file_name2)

    k = cv2.waitKey(frame_period) & 0xFF
    if k == ord('s'):
        if to_make:
            data.append([*last_state, *last_state2, *to_make[0]])
            to_make.pop(0)
        else:
            ne = np.array(data)
            df = pd.DataFrame(ne, columns=['x', 'y', 'r', 'x2', 'y2', 'r2', 'x_real', 'y_real', 'z_real'])
            df.to_csv('camera_dots.csv', index=False)
            print(df)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
sys.exit(app.exec_())