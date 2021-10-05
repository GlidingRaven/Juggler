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
options_window = 'Options'
before_window = 'Before'
after_window = 'After'
original_window = 'Original and result'
color_ranges = (12,158,176),(27,255,255)#((0,74,163),(82,255,255)) #((12,156,176),(24,255,255))

cap = cv2.VideoCapture(file_name)
frame_period = 30
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
kwargs = {'minDist': 200,
          'param1': 80,
          'param2': 80,
          'minRadius': 10,
          'maxRadius': 150}
blur_n = [3,3]
kernel_o = np.ones((3,3),np.uint8)
kernel_c = np.ones((9,9),np.uint8)
alpha = 0.5 # for filter

def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
def ChangeBlur(value):
    global blur_n
    if value%2 == 1: blur_n = [value,value]
def ChangeKernelOpen(value):
    global kernel_o
    if value%2 == 1: kernel_o = np.ones((value,value),np.uint8)
def ChangeKernelClose(value):
    global kernel_c
    if value%2 == 1: kernel_c = np.ones((value,value),np.uint8)
def ChangeDist(value): kwargs['minDist'] = value if value != 0 else 1
def Change1(value): kwargs['param1'] = value if value != 0 else 1
def Change2(value): kwargs['param2'] = value if value != 0 else 1
def ChangeMinRadius(value): kwargs['minRadius'] = value if value != 0 else 1
def ChangeMaxRadius(value): kwargs['maxRadius'] = value if value != 0 else 1
def ChangeAlpha(value):
    global alpha
    alpha = value/10

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

myGUI = Plotter.CustomMainWindow(queue_size = 800, y_limit = 150, start_offset = 75, ylabel='radius')
mySrc = Plotter.Communicate()
mySrc.data_signal.connect(myGUI.addData_callbackFunc)

x_cord, y_cord, radius = 0, 0, 0

while True:
    ret, frame = cap.read()

    if ret:
        blur = cv2.GaussianBlur(frame, tuple(blur_n), 0)  # blure
        output = blur.copy()
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # to HSV
        ready = cv2.inRange(hsv, color_ranges[0], color_ranges[1])  # Color filter
        cv2.imshow(before_window, ready)
        ready = cv2.morphologyEx(ready, cv2.MORPH_OPEN, kernel_o)  # Open filter
        ready = cv2.morphologyEx(ready, cv2.MORPH_CLOSE, kernel_c)  # Close filter
        # gray = cv2.cvtColor(ready, cv2.COLOR_BGR2GRAY)
        # gray3 = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        cv2.imshow(after_window, ready)
        # print(len(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[0][0]))

        circles = cv2.HoughCircles(ready, cv2.HOUGH_GRADIENT, 2, **kwargs)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            radius = circles[0][2] * (1 - alpha) + radius * alpha # exponential moving average filter
            x_cord, y_cord = circles[0][0], circles[0][1]
            mySrc.data_signal.emit(radius)
            output = cv2.putText(output, str(len(circles)), (50, 50), font, 1, color, 1)
        else:
            output = cv2.putText(output, '.', (50, 50), font, 1, color, 1)
            mySrc.data_signal.emit(0)
        cv2.imshow(original_window, output)

        def slised_window(x_cord, y_cord, radius):
            x0 = int(x_cord - radius)
            x1 = int(x_cord + radius)
            y0 = int(y_cord - radius)
            y1 = int(y_cord + radius)

            if (0 < x0 < 640) and (0 < x1 < 640) and (0 < y0 < 480) and (0 < y1 < 480):
                cropped = frame[y0:y1, x0:x1]
                # print(x0,x1,y0,y1)
                # print(cropped)
                cropped = cv2.resize(cropped, (200, 200))
                cv2.imshow('23', cropped)

        slised_window(x_cord, y_cord, radius)
    else:
        cap = cv2.VideoCapture(file_name)


    k = cv2.waitKey(frame_period) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
sys.exit(app.exec_())