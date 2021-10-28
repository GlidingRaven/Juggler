# This file Contains classes for Simple Plotter (CustomMainWindow) and Ball Location Screen (Location_screen)
import random, math, time, sys, os, functools, time, threading, cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import random as rd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# This Class is modified version of [This](https://stackoverflow.com/questions/11874767) code
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


class Location_screen():
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX

        self.view_above = np.zeros((400, 400))
        cv2.rectangle(self.view_above, (200 - 150, 200 - 150), (200 + 150, 200 + 150), 0.5, 1)  # platform
        self.view_side = np.full((400, 100), 0.1)
        self.draw_scale(self.view_side, 50, 0.5)
        self.history = []
        self.last_cord = [0, 0]

    def draw_scale(self, matrix, x_line, color):
        text_scale = 0.4
        cv2.line(matrix, (x_line, 30), (x_line, 370), color, 1)
        for i in range(0, 41, 2):
            y_cor = 400 - 30 - i * 8

            if i % 10 == 0:
                line_width = 10
                cv2.putText(matrix, str(i), (x_line + 15, y_cor), self.font, text_scale, color, 1)
            else:
                line_width = 5

            cv2.line(matrix, (x_line - line_width, y_cor), (x_line + line_width, y_cor), color, 1)

    def draw_cross(self, matrix, cord, size, color):
        font_size = 0.6
        text_line_y_cord = 380
        cv2.rectangle(matrix, (200 + int(cord[0]), 200 + int(cord[1])), (200 + int(cord[0]), 200 + int(cord[1])), color,
                      size)
        cv2.putText(matrix, str(round(cord[0]/10, 1)), (50, text_line_y_cord), self.font, font_size, color, 1)  # draw x-cord
        cv2.putText(matrix, str(round(cord[1]/10, 1)), (140, text_line_y_cord), self.font, font_size, color, 1)  # draw y-cord

    def draw_arrow(self, matrix, cord, size, color):
        font_size = 0.5
        line_width = 12
        # text_line_y_cord = 380
        y_cor = int((400 - 30 - cord[2] * 8 / 10))
        cv2.line(matrix, (50 - line_width, y_cor), (50 + line_width, y_cor), color, size)
        cv2.putText(matrix, str(round(cord[2]/10, 1)), (30, 390), self.font, font_size, color, 1)  # draw z-cord

    def draw_history(self, matrix, history, size, color):
        # print(history)
        for hist in history:
            cv2.circle(matrix, (hist[0] + 200, hist[1] + 200), size, color)
            # cv2.line(matrix, (50 - line_width, y_cor), (50 + line_width, y_cor), color, size)

    def draw_vector(self, matrix, vec, size, color):
        start = []
        font_size = 0.5
        font_color = (255, 255, 255)
        text_line_y_cord = 380
        start.append(self.last_cord[0] + 200)
        start.append(self.last_cord[1] + 200)
        vec_int = cord_int = np.round(vec).astype(int)
        cv2.putText(matrix, str(vec_int[0]), (200, text_line_y_cord), self.font, font_size, font_color, 1)  # draw vector values
        cv2.putText(matrix, str(vec_int[1]), (280, text_line_y_cord), self.font, font_size, font_color, 1)
        vector = np.multiply(vec_int, 10)
        cv2.line(matrix, (start[0], start[1]), (start[0] + vector[0], start[1] + vector[1]), color, size)


    def make_screen(self, cord, history_size = 0, resize=False, size=(1000,800)):
        self.last_cord = cord
        view_above_now = np.copy(self.view_above)
        view_side_now = np.copy(self.view_side)

        if history_size > 0:
            self.history.append(cord)
            if len(self.history) > history_size: self.history.pop(0)
            self.draw_history(view_above_now, self.history, 1, 0.5)

        self.draw_cross(view_above_now, cord, 5, 1)
        self.draw_arrow(view_side_now, cord, 1, 1)

        for_show = np.hstack((view_above_now, view_side_now))
        if resize:
            for_show = cv2.resize(for_show, size)

        return for_show