from __future__ import print_function
import numpy as np
import random
import cv2
import os
from six.moves import cPickle

"""
def process(filename, label):

    random_degree = random.uniform(-10, 10)
    shear_degree = random.uniform(-25, 25)
    x_offset = random.uniform(-5, 5)
    y_offset = random.uniform(-5, 5)
    zoom = random.uniform(1, 1.5)
    rows, cols = img.shape

    tt = np.tan(shear_degree * np.pi / 180)
    M1 = np.array([[1, 0, 0] ,[tt, 1, -32 * tt]], np.float32)
    M2 = cv2.getRotationMatrix2D((cols/2, rows/2), random_degree, 1)
    M2[0][2] += x_offset
    M2[1][2] += y_offset
    img = cv2.warpAffine(img, M1[:2,], (cols, rows))
    img = cv2.warpAffine(img, M2[:2,], (cols, rows))
    if (random.randint(0, 3) == 0):
        img = cv2.Sobel(img, cv2.CV_8U, 1 ,0 ,ksize=3)
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
"""


def readpng():
    if os.path.isfile('data/pickle'):
        f = open('data/pickle', 'rb')
        ret = cPickle.load(f)
        f.close()
        return ret
    else:
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        X_train = np.zeros((35000, 1, 32, 32), dtype='uint8')
        Y_train = np.zeros((35000, ), dtype='uint8')
        X_test = np.zeros((7000, 1, 32, 32), dtype='uint8')
        Y_test = np.zeros((7000, ), dtype='uint8')
        for i in ['Bad/', 'Font/', 'Good/', 'Hnd/']:
            for j in range(11, 37):
                for k in os.listdir('data/{0}Sample{1:03}/'.format(i, j)):
                    if os.path.splitext(k)[-1] != '.png':
                        continue
                    filename = 'data/{0}Sample{1:03}/{2}'.format(i, j, k)
                    print(filename)
                    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (32, 32))
                    if cnt % 6 == 0:
                        X_test[test_cnt, :, :, :] = img
                        Y_test[test_cnt] = j-11
                        test_cnt += 1
                    else:
                        X_train[train_cnt, :, :, :] = img
                        Y_train[train_cnt] = j-11
                        train_cnt += 1
                    cnt += 1
        Y_train = np.reshape(Y_train, (len(Y_train), 1))
        Y_test = np.reshape(Y_test, (len(Y_test), 1))
        f = open('data/pickle', 'wb')
        cPickle.dump([(X_train[:train_cnt, :, :, :], Y_train[:train_cnt]), (X_test[:test_cnt, :, :, :], Y_test[:test_cnt])]
            , f
            , protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        return (X_train[:train_cnt, :, :, :], Y_train[:train_cnt]), (X_test[:test_cnt, :, :, :], Y_test[:test_cnt])

if __name__ == '__main__':
    readpng()
