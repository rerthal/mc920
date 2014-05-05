import classifier
import cv2
import numpy as np
import glob

if __name__ == '__main__':
    for fingerprint in glob.glob("Fingerprints/*.tif"):
        w = 16
        src = cv2.imread(fingerprint)
        src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        filtered = np.zeros(src.shape)
        for i in range(0, src.shape[0] / w):
            for j in range(0, src.shape[1] / w):
                slice = np.array(src)[i*w : (i+1)*w, j*w:(j+1) * w]
                if classifier.is_foreground(slice):
                    for ki in range(0, w):
                        for kj in range(0, w):
                            filtered[i*w + ki][j*w + kj] = slice[ki][kj]
        cv2.imwrite(fingerprint.split('.')[0] + '_filtered.jpg', filtered)
        print fingerprint
