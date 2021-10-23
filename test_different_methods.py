import numpy as np
import cv2, time, sys, os

file_name = 'cam/0b.avi'
color_ranges = (9,171,108),(37,255,255)

cap = cv2.VideoCapture(file_name)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('1')

frame_period = 30
def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
cv2.createTrackbar('FPS', '1', 30, 400, ChangeFPS)

default = {'minDist': 200,
           'param1': 80,
           'param2': 13,
           'minRadius': 10,
           'maxRadius': 150}

while True:
    ret, frame = cap.read()

    if ret:
        blur = cv2.GaussianBlur(frame, tuple([3,3]), 0)  # blure
        output = blur.copy()
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # to HSV
        ready = cv2.inRange(hsv, color_ranges[0], color_ranges[1])  # Color filter
        ready = cv2.morphologyEx(ready, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))  # Open filter
        # ready = cv2.morphologyEx(ready, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))  # Close filter
        circles = cv2.HoughCircles(ready, cv2.HOUGH_GRADIENT, 2, **default)

        # output = cv2.putText(output, '.', (50, 50), font, 1, (255, 255, 255), 1)
        rez = cv2.cvtColor(ready, cv2.COLOR_GRAY2RGB)

        if circles is not None:
            print(circles)
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(rez, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        rez = np.hstack((rez, output))
        # rez = cv2.resize(rez, (640, 240))
        cv2.imshow('HoughCircles', rez)

        contours, hierarchy = cv2.findContours(ready, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rez2 = cv2.cvtColor(ready, cv2.COLOR_GRAY2RGB)
        if len(contours) != 0:
            # cv2.drawContours(rez2, contours, -1, 255, 3)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            (circle_x, circle_y), circle_r = cv2.minEnclosingCircle(c)
            # print((circle_x, circle_y), circle_r)
            if int(circle_r) > 1:
                # cv2.rectangle(rez2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(rez2, (int(circle_x), int(circle_y)), int(circle_r), (0, 255, 0), 2)


        cv2.imshow('Contours', rez2)


    else:
        cap = cv2.VideoCapture(file_name)

    k = cv2.waitKey(frame_period) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()