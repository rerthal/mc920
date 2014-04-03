import glob
import numpy as np
import cv2
import math

def slice(image, W=8):
    lines   = image.shape[0] / W
    columns = image.shape[1] / W
    for i in range(lines):
        for j in range(columns):
            sub_image = [line[i*W: (i+1)*W] for line in image[j*W: (j+1)*W]]
            if len(sub_image) > 0:
                yield (sub_image, i, j)

def max_coord(fourier):
    max = (0,0,0)
    for i in range(1,len(fourier)):
        for j in range(1,len(fourier[i])):
            norm = (fourier[i][j].real * fourier[i][j].real) + (fourier[i][j].imag * fourier[i][j].imag)
            if norm > max[2]:
                max = (i,j,norm)
    return max

fingerprints = glob.glob("Fingerprints/*.tif")
for fingerprint in fingerprints:
    image = cv2.imread(fingerprint)
    image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    for region in slice(image):
        coords = max_coord(np.fft.fft2(region[0]))
        if coords[0] > 0:
            print math.atan(coords[1] / coords[0]), coords[2]
