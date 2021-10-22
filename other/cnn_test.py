import cv2
import numpy as np
import pandas as pd
import time, sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Plotter
from skimage.draw import circle_perimeter_aa
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from cnn.network import Net

file_name = 'cam/ball3_zoom.avi'
cap = cv2.VideoCapture(file_name)
options_window = 'Options'
frame_period = 30

def ChangeFPS(value):
    global frame_period
    if value > 0: frame_period = value

cv2.namedWindow(options_window)
cv2.resizeWindow(options_window, 700, 400);
cv2.createTrackbar('FPS', options_window, 30, 300, ChangeFPS)

def find_circle(img):
    model = Net()
    checkpoint = torch.load('cnn/model.pth.tar',map_location ='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        image = np.expand_dims(np.asarray(img), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize (image)
        image = image.unsqueeze(0)
        output = model(image)

    return [round(i) for i in (200*output).tolist()[0]]


while True:
    ret, frame = cap.read()
    if ret:
        output = frame.copy()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_n = cv2.normalize(src=img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_n = cv2.resize(img_n, (266, 200))
        img_n = img_n[0:200, 30:230]
        cv2.imshow('1', img_n)
        # print(len(img_n), len(img_n[0]))
        circles = find_circle(img_n)
        print(circles)
        # circles = None

        if circles is not None:
            # circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles
            cv2.circle(img_n, (int(x), int(y)), int(abs(r)), (0, 255, 0), 2)
            cv2.imshow('2', img_n)
            cv2.circle(output, (int(30 + x*2.4), int(y*2.4)), int(abs(r*2.4)), (0, 255, 0), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow('3', output)
    else:
        cap = cv2.VideoCapture(file_name)

    k = cv2.waitKey(frame_period) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()