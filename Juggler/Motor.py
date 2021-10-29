import serial, time, sys

class Motor():
    def __init__(self, port = "COM6", speed = 9600):
        try:
            self.ser = serial.Serial(port, speed)
        except Exception:
            print('Motor connection problem')

    def send(self, a1, a2, a3, a4, delay):
        try:
            to_send = "<{3}, {2}, {1}, {0}, {4}>\0".format(int(a1*10), int(a2*10), int(a3*10), int(a4*10), int(delay))
            self.ser.write(bytes(to_send, encoding='utf8'))
            # time.sleep(0.01)
        except Exception:
            print('yay: ', a1, a2, a3, a4, delay)
            print('Motor not exist (send fun)')


    def close(self):
        try:
            self.ser.close()
        except Exception:
            print('Motor not exist (close fun)')

