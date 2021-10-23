#Old version with HoughCircles
import cv2
import numpy as np
import pandas as pd
import time, sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Configurator

file_name = '3a.avi'
color_ranges = (12,158,176),(27,255,255)

cap = cv2.VideoCapture(file_name)

font = cv2.FONT_HERSHEY_SIMPLEX

x_cord, y_cord, radius = 0, 0, 0

while True:
    ret, frame = cap.read()

    if ret:
        blur = cv2.GaussianBlur(frame, tuple(Configurator.blur_n), 0)  # blure
        output = blur.copy()
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # to HSV
        ready = cv2.inRange(hsv, color_ranges[0], color_ranges[1])  # Color filter
        cv2.imshow('Before', ready)
        ready = cv2.morphologyEx(ready, cv2.MORPH_OPEN, Configurator.kernel_o)  # Open filter
        ready = cv2.morphologyEx(ready, cv2.MORPH_CLOSE, Configurator.kernel_c)  # Close filter
        cv2.imshow('After', ready)

        circles = cv2.HoughCircles(ready, cv2.HOUGH_GRADIENT, 2, **Configurator.kwargs)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            radius = circles[0][2] * (1 - Configurator.alpha) + radius * Configurator.alpha # exponential moving average filter
            x_cord, y_cord = circles[0][0], circles[0][1]
            output = cv2.putText(output, str(len(circles)), (50, 50), font, 1, (255, 255, 255), 1)
        else:
            output = cv2.putText(output, '.', (50, 50), font, 1, (255, 255, 255), 1)
        cv2.imshow('Original and result', output)

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


    k = cv2.waitKey(Configurator.frame_period) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()