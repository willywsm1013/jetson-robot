from __future__ import print_function
from threading import Thread
from Queue import Queue
from time import sleep
import serial
import numpy as np

class Arduino(Thread):
    def __init__(self):
        Thread.__init__(self)
        try:
            # self.dev = serial.Serial('/dev/cu.usbmodem14141')
            self.dev = serial.Serial('/dev/ttyACM0')
        except:
            # dev = serial.Serial('/dev/ttyACM1')
            pass
        self.q = Queue(maxsize=5)
        # self.dev.setDTR(False)
        sleep(3)
        self.dev.flushInput()
        self.sleeping = False
        print("Arduino ready")

    def write(self, command):
        # print(command)
        self.dev.write(command)

    def push(self, command):
        self.q.put(command)

    def available(self):
        return self.q.qsize() < 5

    def terminate(self):
        self.push('s 0,0\n')
        self.push('q\n')

    def run(self):
        while True:
            command = self.q.get()
            if command == 'q\n':
                break
            self.write(command)
            sleep(0.01)
