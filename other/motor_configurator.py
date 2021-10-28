import tkinter as tk
from flask import Flask, request
import serial, time, sys

def connect():
    print("connect fun")
    global ser
    port = "COM6"
    bod = 9600
    ser = serial.Serial(port, bod)


def send():
    print("send fun")
    global ser
    a = alpha.get()
    b = beta.get()
    k11 = k1.get() / 100
    k22 = k2.get() / 100
    a1 = int(( base.get() + a*k11 - b*k22 ) * 10)
    a2 = int(( base.get() + a*k11 + b*k22 ) * 10)
    a3 = int(( base.get() - a*k11 + b*k22 ) * 10)
    a4 = int(( base.get() - a*k11 - b*k22 ) * 10)
    delayy = delay.get()
    print(a1, a2, a3, a4)
    to_send = "<{3}, {2}, {1}, {0}, {4}>\0".format(a1, a2, a3, a4, delayy)
    ser.write(bytes(to_send, encoding='utf8'))
    time.sleep(0.01)


def close():
    global ser
    print("close fun")
    ser.close()

# def show_values():
#     print (w1.get(), w2.get())

window = tk.Tk()
base = tk.Scale(window, label = "base", length = 400, from_=0, to=40, orient=tk.HORIZONTAL)
alpha = tk.Scale(window, label = "alpha", length = 400, from_=-30, to=30, orient=tk.HORIZONTAL)
beta = tk.Scale(window, label = "beta", length = 400, from_=-30, to=30, orient=tk.HORIZONTAL)
k1 = tk.Scale(window, label = "k1", length = 400, from_=0, to=100, orient=tk.HORIZONTAL)
k2 = tk.Scale(window, label = "k2", length = 400, from_=0, to=100, orient=tk.HORIZONTAL)
delay = tk.Scale(window, label = "delay", length = 400, from_=50, to=900, orient=tk.HORIZONTAL)

base.set(20)
alpha.set(0)
beta.set(0)
k1.set(5)
k2.set(5)
delay.set(600)

base.pack()
alpha.pack()
beta.pack()
# label = tk.Label(text="Koefs:")# label.pack()
k1.pack()
k2.pack()
delay.pack()

tk.Button(window, text='connect', command=connect).pack()
tk.Button(window, text='close', command=close).pack()
tk.Button(window, text='send', command=send).pack()

window.mainloop()