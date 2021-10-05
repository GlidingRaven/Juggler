# This code is modified version of [This](https://stackoverflow.com/questions/11874767) code
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import functools
import numpy as np
import random as rd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading


class CustomMainWindow(QMainWindow):
    def __init__(self, title = "Plotter", queue_size = 200, y_limit = 100, start_offset = 50, ylabel='data'):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400) # x,y, width, height
        self.setWindowTitle(title)
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        self.myFig = CustomFigCanvas(queue_size=queue_size, y_limit=y_limit, start_offset=start_offset, ylabel=ylabel)
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))
        self.show()
        return

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
        return


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self, queue_size, y_limit, start_offset, ylabel):
        self.addedData = []
        print(matplotlib.__version__)
        # The data
        self.xlim = queue_size
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        self.y = (self.n * 0.0) + start_offset
        # The window
        self.fig = Figure(figsize=(5,5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel(ylabel)
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=2)
        self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(0, y_limit)
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])
        return

    def addData(self, value):
        self.addedData.append(value)
        return

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])

        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]
        return


class Communicate(QObject):
    data_signal = pyqtSignal(float)
