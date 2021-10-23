# See details in other/test_different_methods.py
import numpy as np
import cv2

def prepare_frame(frame, ranges, blur_n = 3, open_filter = 3):
    res = frame.copy()
    res = cv2.GaussianBlur(res, tuple([blur_n,blur_n]), 0)  # blure
    res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)  # to HSV
    res = cv2.inRange(res, ranges[0], ranges[1])  # Color filter
    return cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((open_filter, open_filter), np.uint8))  # Open filter

def find_circle(b_w_frame, min_r = 1):
    contours, hierarchy = cv2.findContours(b_w_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea) # Find biggest
        x, y, w, h = cv2.boundingRect(c)
        (circle_x, circle_y), circle_r = cv2.minEnclosingCircle(c)

        if int(circle_r) > min_r:
            return [[[int(circle_x), int(circle_y), int(circle_r)]]]
        else:
            return None

def draw_circles(frame, circles):
    if circles is not None:
        for (x, y, r) in circles[0]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

class Avg_value(): # Moving average value
    def __init__(self, size = 30):
        self.array = [0]
        self.size = size

    def add(self, val):
        if len(self.array) >= self.size:
            self.array.pop(0)
        self.array.append(val)

    def get(self):
        return sum(self.array) / len(self.array)

class Many_avg_values(): # Moving average for many values, for example: (x,y,z)
    def __init__(self, name_list, size_list):
        self.kwargs = {}
        for i, name in enumerate(name_list):
            self.kwargs[name] = Avg_value(size_list[i])

    def add(self, name, val):
        self.kwargs[name].add(val)

    def get(self, name):
        return self.kwargs[name].get()
