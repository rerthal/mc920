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

if __name__ == '__main__':
    for fingerprint in glob.glob("Fingerprints/*.tif"):
        w = 16
        src = cv2.imread(fingerprint)
        src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        mean, vari, cohe = np.zeros(src.shape), np.zeros(src.shape), np.zeros(src.shape)
        for i in range(0, src.shape[0] / w):
            for j in range(0, src.shape[1] / w):
                slice = np.array(src)[i*w : (i+1)*w, j*w:(j+1) * w]
                dscpt = descriptor(slice, w)
                for ki in range(0, w):
                    for kj in range(0, w):
                        mean[i*w + ki][j*w + kj], vari[i*w + ki][j*w + kj], cohe[i*w + ki][j*w + kj] = dscpt
        print fingerprint
        cv2.imwrite(fingerprint.split('.')[0] + '_rgnl.jpg', src)
        cv2.imwrite(fingerprint.split('.')[0] + '_mean.jpg', mean)
        cv2.imwrite(fingerprint.split('.')[0] + '_vari.jpg', vari)
        cv2.imwrite(fingerprint.split('.')[0] + '_cohe.jpg', cohe * 100)