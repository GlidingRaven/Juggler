import cv2
import numpy as np
import pandas as pd
import time, sys, os

file_name = 'cam/ball2.avi'
options_window = 'Options'

cap = cv2.VideoCapture(file_name)
frame_period = 30
blur_n = [3,3]
pam1, pam2 = 70, 100
kwargs = {'minDist': 200,
          'param1': 80,
          'param2': 80,
          'minRadius': 10,
          'maxRadius': 150}

def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
def ChangeBlur(value):
    global blur_n
    if value%2 == 1: blur_n = [value,value]
def ChangeCanny1(value):
    global pam1
    pam1 = value
def ChangeCanny2(value):
    global pam2
    pam2 = value
def ChangeDist(value): kwargs['minDist'] = value if value != 0 else 1
def Change1(value): kwargs['param1'] = value if value != 0 else 1
def Change2(value): kwargs['param2'] = value if value != 0 else 1
def ChangeMinRadius(value): kwargs['minRadius'] = value if value != 0 else 1
def ChangeMaxRadius(value): kwargs['maxRadius'] = value if value != 0 else 1


cv2.namedWindow(options_window)
cv2.resizeWindow(options_window, 700, 400);
cv2.createTrackbar('FPS', options_window, 30, 300, ChangeFPS)
cv2.createTrackbar('Blur', options_window, 3, 31, ChangeBlur)
cv2.createTrackbar('Canny 1', options_window, 50, 300, ChangeCanny1)
cv2.createTrackbar('Canny 2', options_window, 100, 300, ChangeCanny2)
cv2.createTrackbar('Hough minDist', options_window, 200, 1000, ChangeDist)
cv2.createTrackbar('Hough param1', options_window, 80, 100, Change1)
cv2.createTrackbar('Hough param2', options_window, 24, 200, Change2)
cv2.createTrackbar('Hough minRadius', options_window, 10, 100, ChangeMinRadius)
cv2.createTrackbar('Hough maxRadius', options_window, 150, 200, ChangeMaxRadius)

while True:
    ret, frame = cap.read()
    if ret:
        blur = cv2.GaussianBlur(frame, tuple(blur_n), 0)  # blure
        gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        res = cv2.Canny(gray, pam1, pam2)
        output = frame.copy()

        circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT, 2, **kwargs)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow('edges', res)
        cv2.imshow('original', output)
    else:
        cap = cv2.VideoCapture(file_name)

    k = cv2.waitKey(frame_period) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
