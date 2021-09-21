import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
cap = cv2.VideoCapture(2)

font = cv2.FONT_HERSHEY_SIMPLEX
kwargs = {'minDist': 200,
          'param1': 80,
          'param2': 80,
          'minRadius': 10,
          'maxRadius': 50}
blur_n = [3,3]
kernel_o = np.ones((3,3),np.uint8)
kernel_c = np.ones((9,9),np.uint8)
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
cv2.namedWindow('det')
cv2.createTrackbar('Blur', 'det', 3, 31, ChangeBlur)
cv2.createTrackbar('Open kernel', 'det', 3, 31, ChangeKernelOpen)
cv2.createTrackbar('Close kernel', 'det', 9, 31, ChangeKernelClose)
cv2.createTrackbar('Hough minDist', 'det', 200, 1000, ChangeDist)
cv2.createTrackbar('Hough param1', 'det', 80, 100, Change1)
cv2.createTrackbar('Hough param2', 'det', 80, 200, Change2)
cv2.createTrackbar('Hough minRadius', 'det', 10, 100, ChangeMinRadius)
cv2.createTrackbar('Hough maxRadius', 'det', 50, 200, ChangeMaxRadius)
    
while True:
    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame,tuple(blur_n),0) # blure
    output = blur.copy()
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # to HSV
    ready = cv2.inRange(hsv,(12,156,154),(24,255,255)) # Color filter
    cv2.imshow('1', ready)
    ready = cv2.morphologyEx(ready, cv2.MORPH_OPEN, kernel_o) # Open filter
    ready = cv2.morphologyEx(ready, cv2.MORPH_CLOSE, kernel_c) # Close filter
    cv2.imshow('2', ready)
    
    circles = cv2.HoughCircles(ready, cv2.HOUGH_GRADIENT, 2, **kwargs)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #cv2.imshow('det', np.hstack((ready, output)))
    cv2.imshow('det', output)
    #print(kwargs)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()