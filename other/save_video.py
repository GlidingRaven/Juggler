import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(3)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('3a.avi', fourcc, 30.0, (640, 480))
out2 = cv2.VideoWriter('3b.avi', fourcc, 30.0, (640, 480))
    
while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    
    if ret and ret2:
        cv2.imshow('1', np.hstack((frame,frame2)))
        out.write(frame)
        out2.write(frame2)
    else:
        print(ret,ret2)
        cap.release()
        cap2.release()
        break


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cap.release()
cap2.release()
cv2.destroyAllWindows()