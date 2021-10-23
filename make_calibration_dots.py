import numpy as np
import pandas as pd
import cv2, time, sys, os
from Juggler import Plotter, Ball_detection, Configurator

file_name_1 = 'cam/3a.avi'
file_name_2 = 'cam/3b.avi'
color_ranges_1 = (12,83,194),(35,193,255)
color_ranges_2 = (10,140,143),(26,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX

cap1 = cv2.VideoCapture(file_name_1)
cap2 = cv2.VideoCapture(file_name_2)

to_make = []
data = []
for z in range(0,31,100):
    for y in range(-15,16,150):
        for x in range(-15,16,15):
            to_make.append((x,y,z))
to_make.append((7,7,0))
to_make.append((-7,-7,20))
to_make.append((0,0,30))

window = 30
averages = Ball_detection.Many_avg_values(['x1', 'y1', 'r1', 'x2', 'y2', 'r2'], [window,window,window,window,window,window])

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        if to_make:
            frame1 = cv2.putText(frame1, str(to_make[0]), (50, 50), font, 1, (255,128,128), 2)
        mid1 = Ball_detection.prepare_frame(frame1, color_ranges_1)
        mid2 = Ball_detection.prepare_frame(frame2, color_ranges_2)
        circles1 = Ball_detection.find_circle(mid1)
        circles2 = Ball_detection.find_circle(mid2)

        # cv2.imshow('2', np.hstack((mid1, mid2)))
        if circles1 is not None and circles2 is not None:
            averages.add('x1', circles1[0][0][0]) # badly
            averages.add('y1', circles1[0][0][1])
            averages.add('r1', circles1[0][0][2])
            averages.add('x2', circles2[0][0][0])
            averages.add('y2', circles2[0][0][1])
            averages.add('r2', circles2[0][0][2])
            circle1 = [[[int(averages.get('x1')),int(averages.get('y1')),int(averages.get('r1'))]]]
            circle2 = [[[int(averages.get('x2')),int(averages.get('y2')),int(averages.get('r2'))]]]
            # circle1 = np.array(circle1).astype(int)
            Ball_detection.draw_circles(frame1, circle1)
            Ball_detection.draw_circles(frame2, circle2)

        cv2.imshow('1', np.hstack((frame1, frame2)))
    else:
        cap1 = cv2.VideoCapture(file_name_1)
        cap2 = cv2.VideoCapture(file_name_2)

    k = cv2.waitKey(Configurator.frame_period) & 0xFF
    if k == ord('s'):
        if to_make:
            circle1 = [int(averages.get('x1')), int(averages.get('y1')), int(averages.get('r1'))]
            circle2 = [int(averages.get('x2')), int(averages.get('y2')), int(averages.get('r2'))]
            data.append([*circle1, *circle2, *to_make[0]])
            to_make.pop(0)
        else:
            ne = np.array(data)
            df = pd.DataFrame(ne, columns=['x', 'y', 'r', 'x2', 'y2', 'r2', 'x_real', 'y_real', 'z_real'])
            df.to_csv('data/calibration_dots.csv', index=False)
            print('CSV Saved')
    if k == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
sys.exit(app.exec_())