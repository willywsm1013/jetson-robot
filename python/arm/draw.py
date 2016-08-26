from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

c = {}

for i in np.arange(0, np.pi/2, 0.01):
    for j in np.arange(0, np.pi/2, 0.01):
        xx = 1650*np.cos(i) + 770*np.sin(i+j)
        yy = 1650*np.sin(i) - 770*np.cos(i+j)
        if int(yy) in c:
            if int(xx) < c[int(yy)][0]:
                c[int(yy)][0] = xx
            elif int(xx) > c[int(yy)][1]:
                c[int(yy)][1] = xx
        else:
            c[int(yy)] = [xx, xx]
        x += [xx]
        y += [yy]

m = 0
for key in c:
    if c[key][1] - c[key][0] > m:
        m = c[key][1] - c[key][0]
        idx = key
print(idx, m)

plt.scatter(x, y, alpha=0.5)
plt.show()
