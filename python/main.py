from __future__ import print_function
import cv2
import numpy as np
import tty
import sys
import select
import termios
from car.car import Car
from arduino import Arduino
from time import sleep

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setraw(fd)

arduino = Arduino()
car = Car(arduino)
arduino.start()

def terminate():
    arduino.terminate()
    arduino.join()
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sys.exit()

def Usage():
    print('Usage : BlackLaneDetector <cvp> <source>')
    print('    c : read data from camera')
    print('    v : read data from video')
    print('    p : read data from picture')
    terminate()

"""
car.forward(0)
sleep(2)
car.stop()
"""

def key(com):
    if com == '\x1b[A':
        car.setSpeed(20, 20)
        # print("up")
    elif com == '\x1b[B':
        car.setSpeed(-20, -20)
        # print("down")
    elif com == '\x1b[C':
        car.setSpeed(-20, 20)
        # print("right")
    elif com == '\x1b[D':
        car.setSpeed(20, -20)
        # print("left")
    elif com == ' ':
        car.stop()
    elif com == 'q':
        terminate()
    elif com == 'c':
        tty.setcbreak(sys.stdin.fileno())

while True:
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch = ch + sys.stdin.read(2)
        key(ch)

"""
if len(sys.argv) != 3:
	Usage()
if sys.argv[1]== 'c':
	dev = int(sys.argv[2])
	cap = cv2.VideoCapture(dev)
elif sys.argv[1] == 'v':
	cap = cv2.VideoCapture(sys.argv[2])
elif sys.argv[1] == 'p':
    print 'Not yet implement'
    terminate()
else:
	Usage()

# detector = BlackLaneDetector()

while(True):
    ret, frame = cap.read()
    adapt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(adapt, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  65, 10)
    # Display the resulting frame
    mask = (adapt == 255)
    # frame = frame[mask]
    cv2.imshow('frame', adapt)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
terminate()
