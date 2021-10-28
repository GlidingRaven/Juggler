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
alpha = 0.12 # for filter
camera_fps = 30 # dont change it for no reason, because speed calculation relies on it


def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
    if value == 400: frame_period = 2000

def ChangeAlpha(value):
    global alpha
    alpha = value/100



class Triggers():
    def __init__(self):
        self.count_for_wait = 3
        self.decay_duration = 0
        self.last_z_vel = 0
        self.z_vel_threshold = 1  # threshold to activate action
        self.wait_mode = False


    def detect_velocity_decay(self, z_vel, fun_to_activate):
        if not self.wait_mode:
            if z_vel > 0 and z_vel < self.last_z_vel:
                self.decay_duration += 1
                self.last_z_vel = z_vel
            else:
                self.decay_duration = 0
                self.last_z_vel = z_vel

            if self.decay_duration == self.count_for_wait:
                self.wait_mode = True

        if self.wait_mode:
            if z_vel <= self.z_vel_threshold:
                self.wait_mode = False
                fun_to_activate()
            else:
                pass

cv2.namedWindow(options_window)
# cv2.resizeWindow(options_window, 700, 400);
cv2.createTrackbar('FPS', options_window, 30, 400, ChangeFPS)
cv2.createTrackbar('Alpha %', options_window, 12, 100, ChangeAlpha)