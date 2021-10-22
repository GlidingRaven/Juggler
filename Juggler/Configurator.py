# Creates window with Trackbars
import cv2, time, sys, os
import numpy as np

default = {'minDist': 200,
           'param1': 80,
           'param2': 13,
           'minRadius': 10,
           'maxRadius': 150}

options_window = 'Options'
frame_period = 30
kwargs = {'minDist': default['minDist'],
          'param1': default['param1'],
          'param2': default['param2'],
          'minRadius': default['minRadius'],
          'maxRadius': default['maxRadius']}
alpha = 0.5 # for filter
blur_n = [3,3]
kernel_o = np.ones((3,3),np.uint8)
kernel_c = np.ones((9,9),np.uint8)

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
cv2.createTrackbar('Open ker', options_window, 3, 31, ChangeKernelOpen)
cv2.createTrackbar('Close ker', options_window, 9, 31, ChangeKernelClose)
cv2.createTrackbar('H.minDist', options_window, default['minDist'], 1000, ChangeDist)
cv2.createTrackbar('H.par1', options_window, default['param1'], 100, Change1)
cv2.createTrackbar('H.par2', options_window, default['param2'], 200, Change2)
cv2.createTrackbar('H.minR', options_window, default['minRadius'], 100, ChangeMinRadius)
cv2.createTrackbar('H.maxR', options_window, default['maxRadius'], 200, ChangeMaxRadius)
cv2.createTrackbar('Alpha (filter)', options_window, 5, 10, ChangeAlpha)