from __future__ import print_function
from keras.models import model_from_json
import cv2
import numpy as np

weights_path = 'cnn_model.hdf5'
model_path = 'cnn_model.json'

model = model_from_json(open(model_path).read())
model.load_weights(weights_path)

def classify_img(image):
    image = cv2.resize(image, (32, 32))
    image = np.reshape(image, (1, 1, 32, 32))
    return model.predict(image).argmax()
    """
    print(chr(model.predict(tmp).argmax() + ord('A')))
    sorted_idx = np.argsort(out['prob'][0])
    for j in range(25, 23, -1):
        print((chr(sorted_idx[j]+65), out['prob'][0][sorted_idx[j]]), end="")
    print()
    # print(chr(out['prob'][0].argmax() + ord('A')), end="")
    # print(out['prob'][0])
    """
