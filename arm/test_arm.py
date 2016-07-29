from arm import Arm
from time import sleep
import numpy as np

arm = Arm()

while True:
    for i in np.arange(0, np.pi, 0.1):
        arm.gotoRZ(1081 + 302 * np.cos(i), 841 + 0 * np.sin(i))
    for i in np.arange(np.pi, 0, -0.1):
        arm.gotoRZ(1081 + 302 * np.cos(i), 841 + 0 * np.sin(i))


