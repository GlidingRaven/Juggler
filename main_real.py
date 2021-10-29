import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random, math, time, sys, pickle
from sklearn import model_selection, datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from Juggler import Plotter, Ball_detection, Configurator, Predictors, Motor
from threading import Timer
import warnings
warnings.filterwarnings('ignore')

enable_action = True
file_name_1 = 'cam/8a.avi'
file_name_2 = 'cam/8b.avi'
color_ranges_1 = (10,73,230),(31,255,255)
color_ranges_2 = (14,137,185),(23,255,255)
cap1 = cv2.VideoCapture(file_name_1)
cap2 = cv2.VideoCapture(file_name_2)

Cord_finder = Predictors.Cord_finder('calibration_models.pickle')
Action_predictor = Predictors.ActionPredictor('action_model.pickle')

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
app = QApplication(sys.argv)
QApplication.setStyle(QStyleFactory.create('Plastique'))
myGUI = Plotter.CustomMainWindow(queue_size = 200, y_limit = 400, start_offset = 50, ylabel='z vel')
mySrc = Plotter.Communicate()
mySrc.data_signal.connect(myGUI.addData_callbackFunc)
last_cord = [0,0,0]
vector_avg = Ball_detection.Many_avg_values(['x_vel', 'y_vel', 'z_vel'], [8, 8, 2])
x_vel, y_vel = 0, 0
location_screen = Plotter.Location_screen()
Trigger = Ball_detection.Triggers()
if enable_action:
    Motor = Motor.Motor()
else:
    Motor = Motor.Motor(port = 'dummy')
cv2.imshow('location', location_screen.make_screen(last_cord))

def hit():
    print('Action:')
    # print(cord, real_vel)
    # print(vector_avg.get('z_vel'))
    x, y, z = cord
    x_vel, y_vel, z_vel = vector_avg.get('x_vel'), vector_avg.get('y_vel'), vector_avg.get('z_vel')
    alpha, beta, vel, delay = Action_predictor.predict([x/1000, y/1000, z/1000, x_vel*240, y_vel*240])
    if z < 10:
        delay = 0
    else:
        delay = round(delay)/240 # convert to second from steps

    # print(alpha, beta)
    alpha /= 2
    beta /= 2
    print(alpha, beta)#, vel, delay)
    if alpha > 5: alpha = 5
    if alpha < -5: alpha = -5
    if beta > 5: beta = 5
    if beta < -5: beta = -5
    k11 = 310/100
    k22 = 375/100
    # print(Configurator.base, alpha, k11, beta, k22)
    a1 = Configurator.base + alpha * k11 - beta * k22
    a2 = Configurator.base + alpha * k11 + beta * k22
    a3 = Configurator.base - alpha * k11 + beta * k22
    a4 = Configurator.base - alpha * k11 - beta * k22

    to_send = [a1, a2, a3, a4, Configurator.conf_z_vel]
    print(to_send)
    t = Timer( Configurator.conf_delay/1000, Motor.send, to_send )
    t.start()
    print('\n')

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
            z_cord = Cord_finder.predict_z(circles1[0][0][0], circles1[0][0][1], circles2[0][0][0], circles2[0][0][1]) # in centimeters
            cord = Cord_finder.predict_x_y(circles1[0][0][0], circles2[0][0][0], z_cord) # in milimeters
            cord_int = np.round(cord).astype(int)
            vector = np.subtract(cord, last_cord)
            last_cord = cord
            # print(cord)
            # print(vector)
            for_show = location_screen.make_screen(cord_int, history_size = 30, resize=False)
            for_show = cv2.cvtColor(np.float32(for_show), cv2.COLOR_GRAY2RGB)
            real_vel = np.multiply(vector, Configurator.camera_fps/1000) # in m/s
            # print(real_vel[2])

            if vector[0] < 15 and vector[1] < 15: # filter out Outliers
                vector_avg.add('x_vel', real_vel[0])
                vector_avg.add('y_vel', real_vel[1])
                vector_avg.add('z_vel', real_vel[2])
                # x_vel = vector[0] * (1 - Configurator.alpha) + x_vel * Configurator.alpha  # exponential moving average filter
                # y_vel = vector[1] * (1 - Configurator.alpha) + y_vel * Configurator.alpha

            Trigger.detect_velocity_decay(cord, vector, hit)

            location_screen.draw_vector(for_show, vector, 1, (6, 60, 60))
            # location_screen.draw_vector(for_show, [vector_avg.get('x_vel'), vector_avg.get('y_vel'), 0], 1, (0, 250, 0))
            location_screen.draw_vector(for_show, [x_vel, y_vel], 1, (0, 0, 250))
            mySrc.data_signal.emit(cord[2])

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
Motor.close()
# sys.exit(app.exec_())