import cv2
import numpy as np

def preprocess_image(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # invert (EMNIST style)
    img = 255 - img

    # resize
    img = cv2.resize(img, (28, 28))

    # normalize
    img = img / 255.0

    # flatten
    img = img.reshape(1, -1)

    return img