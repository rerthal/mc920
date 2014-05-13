import cv2
import math
import os
import numpy as np
from sklearn.svm import SVC

def coh(window):
    sobel_x = cv2.Sobel(window, -1, 1, 0)
    sobel_y = cv2.Sobel(window, -1, 0, 1)
    Gxx, Gyy, Gxy = 0, 0, 0
    for i in range(sobel_x.shape[0]):
        for j in range(sobel_x.shape[1]):
            Gxx = Gxx + sobel_x[i][j] * sobel_x[i][j]
            Gyy = Gyy + sobel_y[i][j] * sobel_y[i][j]
            Gxy = Gxy + sobel_x[i][j] * sobel_y[i][j]
    if Gxx + Gyy != 0:
        return math.sqrt((Gxx - Gyy) ** 2 + 4 * (Gxy ** 2)) / (Gxx + Gyy)
    else:
        return 0

def descriptor(directory):
    result = []
    for fingerprint in os.listdir(directory):
        src = cv2.imread(directory + '/' + fingerprint)
        src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        result.append([np.mean(src), np.var(src), coh(src)])
    return result

for directory in os.listdir("./fingerprints"):
    print directory
    background_data = descriptor("./fingerprints/" + directory + "/slices/foreground")
    foreground_data = descriptor("./fingerprints/" + directory + "/slices/background")
    background_labels = map(lambda e: 0, background_data)
    foreground_labels = map(lambda e: 1, foreground_data)
    X = np.array(background_data + foreground_data)
    y = np.array(background_labels + foreground_labels)
    classifier = SVC()
    classifier.fit(X, y)
    for fingerprint in os.listdir("./fingerprints/" + directory + "/original"):
        src = cv2.imread("./fingerprints/" + directory + "/original/" + fingerprint)
        src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        filtered = np.zeros(src.shape)
        mean, variance, coherence = np.zeros(src.shape), np.zeros(src.shape), np.zeros(src.shape)
        #for i in range(0, src.shape[0] / 16 - 1):
        #    for j in range(0, src.shape[1] / 16 - 1):
        #        slice = np.array(src)[i*16 : (i+1)*16, j*16:(j+1) * 16]
        #        x = [np.mean(slice), np.var(slice), coh(slice)]
        #        for ik in range(16):
        #            for jk in range(16):
        #                mean[i*16 + ik][j*16 + jk] = x[0]
        #                variance[i*16 + ik][j*16 + jk] = x[1]
        #                coherence[i*16 + ik][j*16 + jk] = x[2]
        #cv2.imwrite("./fingerprints/" + directory + "/mean/" + fingerprint, mean)
        #cv2.imwrite("./fingerprints/" + directory + "/variance/" + fingerprint, variance)
        #cv2.imwrite("./fingerprints/" + directory + "/coherence/" + fingerprint, coherence * 100)
        for i in range(2, src.shape[0]-2):
            for j in range(2, src.shape[1]-2):
                slice = np.array(src)[i-2 : i+2, j-2:j+2]
                if classifier.predict([np.mean(slice), np.var(slice), coh(slice)]):
                    filtered[i][j] = 255
        filtered_opening = cv2.morphologyEx(cv2.convertScaleAbs(filtered), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        cv2.imwrite("./fingerprints/" + directory + "/filtered/" + fingerprint, filtered_opening)
        print "./fingerprints/" + directory + "/filtered/" + fingerprint