import cv2, argparse

file_name = 'cam/0a.avi'

max_value = 255
frame_period = 30
max_value_H = 360//2
low_H, low_S, low_V = 12, 156, 176
high_H, high_S, high_V = 24, 255, 255
window_detection_name = 'Object Detection'
low_H_name, low_S_name, low_V_name = 'L Hue', 'L Sat', 'L Val'
high_H_name, high_S_name, high_V_name = 'H Hue', 'H Sat', 'H Val'

def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


cap = cv2.VideoCapture(file_name)
cv2.namedWindow(window_detection_name)
cv2.createTrackbar('FPS', window_detection_name, 30, 100, ChangeFPS)
cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

while True:
    
    ret, frame = cap.read()
    if ret:
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        cv2.imshow('1', frame)
        cv2.imshow(window_detection_name, frame_threshold)
    else:
        cap = cv2.VideoCapture(file_name)

    k = cv2.waitKey(frame_period) & 0xFF
    if k == 27:
        break
print('HSV params: ({},{},{}),({},{},{})'.format(low_H, low_S, low_V, high_H, high_S, high_V))
cap.release()
cv2.destroyAllWindows()