import numpy as np
import math
import cv2
import sys
from pymorph import erode
from pymorph import dilate
import os

def direction(window, alpha=0.0):
    n = len(window)
    test_points = []
    for k in range(-n/2 + 1, n/2 + 1):
        x = int(round(n / 2 + k * math.cos(alpha)))
        y = int(round(n / 2 + k * math.sin(alpha)))
        test_points.append(window[x][y])
    return test_points

def dominant_direction(window, op):
    D = 8
    dominant_direction = None
    dominant_delta = 0.0
    for d in range(D):
        alpha = math.radians(d * 180.0 / float(D))
        beta = math.radians(90 + d * 180.0 / float(D))
        delta = op(direction(window, alpha)) - op(direction(window, beta))
        if delta > dominant_delta:
            dominant_delta = delta
            dominant_direction = d * 180.0 / float(D)
    return dominant_direction

def b(image, w=2, sigma=25, k=15):
    mask = -sigma * np.array([[1,1,1], [1,0,1], [1,1,1]])
    erosion = np.copy(image)
    dilation = np.copy(image)
    for i in range(k):
        erosion = erode(erosion, mask)
        dilation = dilate(dilation, mask)
    filtered = np.copy(image)
    for i in range(w, image.shape[0]-w):
        for j in range(w, image.shape[1]-w):
            slice = np.array(image)[i-w:i+w+1, j-w:j+w+1]
            if (dilation[i][j] - image[i][j] > image[i][j] - erosion[i][j]):
                filtered[i][j] = 0
                alpha = dominant_direction(slice, np.std)
                if alpha != None:
                    alpha = math.radians(alpha + 90)
                    filtered[i][j] = cv2.dilate(np.array(slice, np.uint8), np.array(direction(slice, alpha), np.uint8), iterations = k)[w][0]
    return filtered

def field_detection_block(image, w=22):
    filtered = np.zeros(image.shape)
    for i in range(w, image.shape[0]-w):
        for j in range(w, image.shape[1]-w):
            slice = np.array(image)[i-w:i+w+1, j-w:j+w+1]
            alpha = dominant_direction(slice, op=np.median)
            if alpha != None: filtered[i][j] = 255
    return filtered

def filter(image, mask):
    filtered = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] == 0: filtered[i][j] = 255
            else: filtered[i][j] = image[i][j]
    return filtered


if __name__ == '__main__':
    for directory in os.listdir("./fingerprints"):
        print directory
        for fingerprint in os.listdir("./fingerprints/" + directory):
            filename = "./fingerprints/" + directory + "/" + fingerprint
            src = cv2.imread(filename)
            src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
            background = b(src)
            mask = field_detection_block(src - background)
            filtered = filter(src, mask)
            cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_src.jpg', src)
            cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_background.jpg', background)
            cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_mask.jpg', mask)
            cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_filtered.jpg', filtered)
            print fingerprint