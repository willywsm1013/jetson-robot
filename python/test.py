from __future__ import print_function
import select
import sys
import tty
import termios
from time import sleep

def something(com):
    if com == '\x1b[A':
        print("up")
    elif com == '\x1b[B':
        print("down")
    elif com == '\x1b[C':
        print("right")
    elif com == '\x1b[D':
        print("left")
    elif com == 'q':
        sys.exit()
    elif com == 'c':
        tty.setcbreak(sys.stdin.fileno())

# If there's input ready, do something, else do something
# else. Note timeout is zero so select won't block at all.
tty.setraw(sys.stdin.fileno())
while True:
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch = ch + sys.stdin.read(2)
        something(ch)
