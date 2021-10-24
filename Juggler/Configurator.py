# Creates window with Trackbars
import cv2, time, sys, os
import numpy as np

options_window = '1'

default = {'frame_period': 30,
           'param1': 80,
           'param2': 13,
           'minRadius': 10,
           'maxRadius': 150}

frame_period = default['frame_period']
alpha = 0.5 # for filter
camera_fps = 30 # dont change it for no reason, because speed calculation relies on it

def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
    if value == 400: frame_period = 2000

def ChangeAlpha(value):
    global alpha
    alpha = value/100

cv2.namedWindow(options_window)
# cv2.resizeWindow(options_window, 700, 400);
cv2.createTrackbar('FPS', options_window, 30, 400, ChangeFPS)
cv2.createTrackbar('Alpha %', options_window, 50, 100, ChangeAlpha)