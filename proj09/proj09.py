import glob
import numpy as np
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
    max_i   = 0
    max_j   = 0
    max_n   = 0
    for i in range(1,len(fourier)):
        for j in range(1,len(fourier[i])):
            norm = (fourier[i][j].real * fourier[i][j].real) + (fourier[i][j].imag * fourier[i][j].imag)
            if norm > max_n:
                max_n = norm
                max_i = i
                max_j = j

    return (max_i,max_j)

fingerprints = glob.glob("Fingerprints/*.tif")

for fingerprint in fingerprint:
    for slice in slice(fingerprint):
        coords = max_coord(np.fft.fft2(slice[0]))
        print math.atan(coords[0] / coords[1])
