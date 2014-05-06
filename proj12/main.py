import classifier
import cv2
import numpy as np
import glob

for fingerprint in glob.glob("Fingerprints/*.tif"):
    src = cv2.imread(fingerprint)
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    filtered = np.zeros(src.shape)
    for i in range(2, src.shape[0]-2):
        for j in range(2, src.shape[1]-2):
            slice = np.array(src)[i-2 : i+2, j-2:j+2]
            if classifier.is_foreground(slice, w=4): filtered[i][j] = 255
    filtered_opening = cv2.morphologyEx(cv2.convertScaleAbs(filtered), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    cv2.imwrite(fingerprint.split('.')[0] + '_filtered.jpg', filtered_opening)
    print fingerprint
