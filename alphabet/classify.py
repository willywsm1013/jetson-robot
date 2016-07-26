from __future__ import print_function
import caffe
import cv2
import numpy as np
import png
import itertools
import matplotlib.pyplot as plt
from PIL import Image

caffe.set_mode_cpu()

net = caffe.Net("test3.prototxt", "test3.caffemodel", caffe.TEST)
# print(net.blobs['conv1'].data)
# print(net.params['conv1'][0].data)

for i in range(0, 29):
    if i % 5 == 0:
        print()
    filename = "letters/stop_sign3letter" + str(i) + ".jpg"
    # filename = "image/" + str(i) + ".jpg"
    # im = np.array(Image.open(filename), dtype="float")
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.asarray(image, dtype="float")
    image = (image) / 256
    net.blobs['data'].data[...] = image
    out = net.forward()
    sorted_idx = np.argsort(out['prob'][0])
    for j in range(25, 23, -1):
        print((chr(sorted_idx[j]+65), out['prob'][0][sorted_idx[j]]), end="")
    print()
    # print(chr(out['prob'][0].argmax() + ord('A')), end="")
    # print(out['prob'][0])

"""
for i in range(11, 12):
    a = []
    for j in range(1, 2):
        filename = "test.png"
        # filename = "alphabet/data/Sample{0:03}/img{0:03}-{1:05}.png".format(i, j) 
        im = np.array(Image.open(filename), dtype="float")/256
        a.append(im[np.newaxis, : , :])
        # net.blobs['data'].data[...] = im[np.newaxis, np.newaxis, : , :]
        # print(chr(out['prob'].argmax() + ord('A')))
    net.blobs['data'].data[...] = np.asarray(a)
    outs = net.forward()
    for out in outs['prob']:
        print(out)
"""
