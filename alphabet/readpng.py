from __future__ import print_function
import numpy as np
import lmdb
import caffe
import random
import cv2
import os

train_env = lmdb.open('train_lmdb', map_size=36*1024*1024*1024)
test_env = lmdb.open('test_lmdb', map_size=24*1024*1024*1024)

def process(filename, label):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    random_degree = random.uniform(-10, 10)
    shear_degree = random.uniform(-25, 25)
    x_offset = random.uniform(-5, 5)
    y_offset = random.uniform(-5, 5)
    zoom = random.uniform(1, 1.5)
    rows, cols = img.shape

    """
    tt = np.tan(shear_degree * np.pi / 180)
    M1 = np.array([[1, 0, 0] ,[tt, 1, -32 * tt]], np.float32)
    M2 = cv2.getRotationMatrix2D((cols/2, rows/2), random_degree, 1)
    M2[0][2] += x_offset
    M2[1][2] += y_offset
    img = cv2.warpAffine(img, M1[:2,], (cols, rows))
    img = cv2.warpAffine(img, M2[:2,], (cols, rows))
    if (random.randint(0, 3) == 0):
        img = cv2.Sobel(img, cv2.CV_8U, 1 ,0 ,ksize=3)
    """
    # img = cv2.resize(img, (64, 64), fx=zoom, fy=zoom, interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img, (64, 64))
    if (random.randint(0, 1) == 0):
        img = 255 - img

    datum = caffe.io.array_to_datum(img[np.newaxis, :, :], label)
    str_id = str(random.randint(0, 10000000))

    if random.random() < 0.15:
        test.put(str_id, datum.SerializeToString())
    else:
        train.put(str_id, datum.SerializeToString())


cnt = 0
with train_env.begin(write=True) as train, test_env.begin(write=True) as test:
    for i in ["Bad/", "Font/", "Good/", "Hnd/"]:
        for j in range(11, 37):
            for k in os.listdir(i + "Sample{0:03}/".format(j)):
                if os.path.splitext(k)[-1] != ".png":
                    continue
                filename = "{0}Sample{1:03}/{2}".format(i, j, k)
                print(filename)
                cnt += 1
                process(filename, j-11)
print(cnt)
