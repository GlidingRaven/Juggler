from flask import Flask, request
import serial, time, sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QFormLayout, QLabel
from PyQt5.QtGui import QIcon, QIntValidator, QFont
from PyQt5.QtCore import pyqtSlot

# <300, 300, 300, 300, 700>
# <1, 2, 3, 4, msec>

class Window():
    def __init__(self):
        self.port = "COM6"
        self.bod = 9600
        self.app = QApplication(sys.argv)
        self.widget = QWidget()
        self.flo = QFormLayout()
        font_size = 14
        font = "Arial"

        class text_box():
            def __init__(self, widget, max_len = 3):
                self.val = 0
                self.box = QLineEdit(widget)
                self.box.setValidator(QIntValidator())
                self.box.setMaxLength(max_len)
                self.box.setFont(QFont(font, font_size))
                self.box.textChanged.connect(self.changed)

            def changed(self, text):
                if text != None: self.val = int(text)
                # print('val = ', type(self.val), self.val)

            def set(self, val):
                self.box.setText(val)

            def get(self):
                return self.val

        self.a1, self.a2, self.a3, self.a4 = text_box(self.widget), text_box(self.widget), text_box(self.widget), text_box(self.widget)
        self.delay = text_box(self.widget, 4)

        connect_button = QPushButton(self.widget)
        connect_button.setText("Connect")
        connect_button.clicked.connect(self.connect)

        send_button = QPushButton(self.widget)
        send_button.setText("Send")
        send_button.clicked.connect(self.send)

        close_button = QPushButton(self.widget)
        close_button.setText("Close")
        close_button.clicked.connect(self.close)

        self.label = QLabel()
        self.label.setText('Oct 2021')

        self.flo.addRow("1", self.a1.box)
        self.flo.addRow("2", self.a2.box)
        self.flo.addRow("3", self.a3.box)
        self.flo.addRow("4", self.a4.box)
        self.flo.addRow("Delay (ms)", self.delay.box)
        self.flo.addRow("", connect_button)
        self.flo.addRow("", send_button)
        self.flo.addRow("", close_button)
        self.flo.addRow("", self.label)

        init_num = '100'
        self.a1.set(init_num)
        self.a2.set(init_num)
        self.a3.set(init_num)
        self.a4.set(init_num)
        self.delay.set('700')

        self.widget.setGeometry(50, 50, 320, 200)
        self.widget.setLayout(self.flo)
        self.widget.setWindowTitle("Juggler motor tester")
        self.widget.show()
        sys.exit(self.app.exec_())

    def a1_changed(self, text):
        if text != None: self.a1 = int(text)
        # print('a1 = ', type(self.a1), self.a1)

    def a2_changed(self, text):
        if text != None: self.a2 = int(text)

    def a3_changed(self, text):
        if text != None: self.a3 = int(text)

    def a4_changed(self, text):
        if text != None: self.a4 = int(text)

    def connect(self):
        print("connect fun")
        self.ser = serial.Serial(self.port, self.bod)

    def send(self):
        print("send fun")
        # time.sleep(2)         self.ser.write(b"<{}, 300, 300, 300, 700>\0".format(self.a1.get()))
        # print(self.a1.get())
        to_send = "<{3}, {2}, {1}, {0}, {4}>\0".format(self.a1.get(), self.a2.get(), self.a3.get(), self.a4.get(), self.delay.get())
        self.ser.write(bytes(to_send, encoding='utf8'))
        time.sleep(0.01)

    def close(self):
        print("close fun")
        self.ser.close()

d = Window()