import cv2
import numpy as np
import pandas as pd
import time, sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Juggler import Plotter, Ball_detection, Configurator

file_name = 'other/cam/1st/ball3_zoom.avi'
color_ranges = (18,164,176),(26,255,255)

cap = cv2.VideoCapture(file_name)

font = cv2.FONT_HERSHEY_SIMPLEX

app = QApplication(sys.argv)
QApplication.setStyle(QStyleFactory.create('Plastique'))
myGUI = Plotter.CustomMainWindow(queue_size = 400, y_limit = 100, start_offset = 0, ylabel='radius')
mySrc = Plotter.Communicate()
mySrc.data_signal.connect(myGUI.addData_callbackFunc)

x_cord, y_cord, radius = 0, 0, 0

while True:
    ret, frame = cap.read()

    if ret:
        output = frame.copy()
        mid1 = Ball_detection.prepare_frame(frame, color_ranges)
        circles = Ball_detection.find_circle(mid1)

        if circles is not None:
            # circles = np.round(circles[0, :]).astype("int")
            circles = circles[0]
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            print(circles)

            radius = circles[0][2] * (1 - Configurator.alpha) + radius * Configurator.alpha # exponential moving average filter
            x_cord, y_cord = circles[0][0], circles[0][1]
            mySrc.data_signal.emit(radius)
            output = cv2.putText(output, str(len(circles)), (50, 50), font, 1, (255, 255, 255), 1)
        else:
            output = cv2.putText(output, '.', (50, 50), font, 1, (255, 255, 255), 1)
            mySrc.data_signal.emit(0)
        cv2.imshow('1', output)

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
sys.exit(app.exec_())