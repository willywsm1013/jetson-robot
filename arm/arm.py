from threading import Thread, Event
from time import sleep
import serial
import numpy as np

class Arm(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.param = ((800, 2125), (775, 2125), (850, 2150), (800, 2100), (800, 2200), (1300, 1800))
        self.now = [90, 90, 90, 90, 90, 90]
        self.event = event
        self.l1 = 850
        self.l2 = 770
        try:
            dev = serial.Serial('/dev/ttyACM0')
        except:
            dev = serial.Serial('/dev/ttyACM1')
        dev.setDTR(False)
        sleep(1)
        dev.flushInput()
        self.dev = dev

    def write(self):
        s = 'h,'
        for i, degree in enumerate(self.now):
            """
            if degree < 0:
                degree = 0
            elif degree > 180:
                degree = 180
            """
            us = self.param[i][0] + (self.param[i][1] - self.param[i][0]) * degree / 180
            s += str(int(us)) + ','

        s = s[:-1] + '\n'
        print(s)
        self.dev.write(s)
        sleep(0.01)

    def rad2deg(self, rad):
        return rad * 180 / np.pi

    def gotoRZ(self, r, z):
        # print(r, z)
        degrees = [90] * 6
        d_square = r**2 + z**2
        d = np.sqrt(d_square)
        # print((self.l1 ** 2 + d_square - self.l2 ** 2) / (2 * self.l1 * d))
        # print((self.l1 ** 2 + self.l2 ** 2 - d_square) / (2 * self.l1 * d))
        degrees[1] = self.rad2deg(np.arctan2(z, r) + np.arccos((self.l1 ** 2 + d_square - self.l2 ** 2) / (2 * self.l1 * d)))
        degrees[2] = self.rad2deg(np.arccos((self.l1 ** 2 + self.l2 ** 2 - d_square) / (2 * self.l1 * self.l2))) - 90
        degrees[3] = 90 - degrees[1] - degrees[2]
        if degrees != self.now:
            self.now = degrees
            self.event.set()

    def run(self):
        while True:
            self.event.wait()
            self.write()
            self.event.clear()

if __name__ == "__main__":
    event = Event()
    arm = Arm(event)
    arm.start();
    #arm.write([90, 90, 90, 90, 90, 90])
    #arm.gotoRZ(1383, 841)
    while True:
        for i in np.arange(0, 2 * np.pi, 0.05):
            arm.gotoRZ(1300+300*np.cos(i), -200+300*np.sin(i))
            sleep(0.01)
