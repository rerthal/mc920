import numpy as np
import cv2
import math
import glob

def coh(window, w=8):
    sobel_x = cv2.Sobel(window, -1, 1, 0)
    sobel_y = cv2.Sobel(window, -1, 0, 1)
    Gxx, Gyy, Gxy = 0, 0, 0
    for i in range(w):
        for j in range(w):
            Gxx = Gxx + sobel_x[i][j] * sobel_x[i][j]
            Gyy = Gyy + sobel_y[i][j] * sobel_y[i][j]
            Gxy = Gxy + sobel_x[i][j] * sobel_y[i][j]
    if Gxx + Gyy != 0:
        return math.sqrt((Gxx - Gyy) ** 2 + 4 * (Gxy ** 2)) / (Gxx + Gyy)
    else:
        return 0

def descriptor(window, w=8):
    return np.mean(window), np.var(window), coh(window, w)