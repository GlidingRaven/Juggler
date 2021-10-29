# Creates window with Trackbars
import cv2, time, sys, os
import numpy as np

options_window = '1opt'

default = {'frame_period': 30,
           'param1': 80,
           'param2': 13,
           'minRadius': 10,
           'maxRadius': 150}

frame_period = default['frame_period']
alpha = 0.12 # for filter
camera_fps = 30 # dont change it for no reason, because speed calculation relies on it
base = 40
conf_z_vel = 200
conf_delay = 200

def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
    if value == 400: frame_period = 2000

def ChangeAlpha(value):
    global alpha
    alpha = value/100

def ChangeBase(value):
    global base
    base = value

def ChangeZVel(value):
    global conf_z_vel
    conf_z_vel = value

def ChangeDelay(value):
    global conf_delay
    conf_delay = value


cv2.namedWindow(options_window)
cv2.resizeWindow(options_window, 600, 300);
cv2.createTrackbar('FPS', options_window, 30, 400, ChangeFPS)
# cv2.createTrackbar('Alpha %', options_window, 12, 100, ChangeAlpha)
cv2.createTrackbar('Base %', options_window, 40, 60, ChangeBase)
cv2.createTrackbar('Z vel %', options_window, 200, 500, ChangeZVel)
cv2.createTrackbar('Delay %', options_window, 200, 800, ChangeDelay)