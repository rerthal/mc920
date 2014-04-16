import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def slice(image, M=8):
    lines   = image.shape[0] / M
    columns = image.shape[1] / M
    for i in range(1,lines):
        for j in range(1,columns):
            sub_image = [line[(i-1)*M: (i+1)*M] for line in image[(j-1)*M: (j+1)*M]]
            if len(sub_image) > 0:
                yield (np.asarray(sub_image), i, j)

def intensity(p):
    return p.real**2 + p.imag**2
intensity_vec = np.vectorize(intensity)

def max_coord(fourier,M=8):
    # shift zero-frequency term to the centre of the array
    fourier = np.fft.fftshift(fourier)
    centre=(M,M)
    max = (0,0,0,0)
    for i in range(1,len(fourier)):
        for j in range(1,len(fourier[i])):
#            norm = (fourier[i][j].real * fourier[i][j].real) + (fourier[i][j].imag * fourier[i][j].imag)
            norm = intensity(fourier[i][j])
            if norm > max[2] and (i,j) != centre:
                dist = math.sqrt((i-centre[0])**2 + (j-centre[1])**2)
                max = (i-centre[0],j-centre[0],norm,dist)
    return max

def remove_ring_points(fourier, r, M=8, W=1):
    fourier = np.fft.fftshift(fourier)
    centre = (M,M)
    circle = np.ones((2*M, 2*M))
    cv2.circle(circle, centre, int(round(r)), color=0)
    return fourier*circle

def remove_dc(fourier, M=8):
    centre = (M,M)
    aux = np.ones((2*M, 2*M))
    aux[(centre)] = 0
    return fourier*aux

def band_pass_filter(fourier, f, W=1):
    return fourier

def gaussian_directional_filter(theta, size=5, sigma=1):
    mask = np.ones((size,size))
    return mask

def remove_percentile(fourier,t=0.05):
    f = intensity_vec(fourier)
    m = -100000
    for i in range(len(f)):
        for j in range(len(f[0])):
            if f[i][j] >= m:
                m = f[i][j]
                print i,j 
    print m
    print np.amax(f)
    hist = np.bincount(f.astype(int).ravel(), minlength=256)
#    hist = cv2.normalize(hist, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    
    return f

def test():
    fingerprints = glob.glob("Fingerprints/1_1.tif")
    for fingerprint in fingerprints:
        image = cv2.imread(fingerprint)
        image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        # We'll print frequency for all images
        i = fingerprint[-5] # get image suffix
        # but we'll print line statistics for only one line of one image
        if fingerprint == 'Fingerprints/1_1.tif':
            f_line = open('line.txt', 'w')
        for region in slice(image):
            if len(region[0]) != 16:
                continue
            i = str(region[1])
            j = str(region[2])
            if len(i) == 1:
                i = '0' + i
            if len(j) == 1:
                j = '0' + j
            # we only want to print a few regions from one image
            #if fingerprint == 'Fingerprints/1_1.tif':
            #    if region[2] == 16 and region[1] in [3,40,20,23,26,30]:
            #        cv2.imwrite('region'+ j +'-'+ i +'.png', region[0])
            fft = np.fft.fft2(region[0])
            coords = max_coord(np.fft.fft2(region[0]))
            d = str(coords[3] if coords[3] < 3 else 0)
            if region[2] == 16 and region[1] == 30:
                x = remove_ring_points(fft, float(d), M=8, W=1)
                x = remove_dc(x,M=8)
                x = remove_percentile(x)
                return x

if __name__ == '__main__':
    fingerprints = glob.glob("Fingerprints/1_1.tif")
    for fingerprint in fingerprints:
        image = cv2.imread(fingerprint)
        image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        # We'll print frequency for all images
        i = fingerprint[-5] # get image suffix
        # but we'll print line statistics for only one line of one image
        if fingerprint == 'Fingerprints/1_1.tif':
            f_line = open('line.txt', 'w')
        for region in slice(image):
            if len(region[0]) != 16:
                continue
            i = str(region[1])
            j = str(region[2])
            if len(i) == 1:
                i = '0' + i
            if len(j) == 1:
                j = '0' + j
            # we only want to print a few regions from one image
            #if fingerprint == 'Fingerprints/1_1.tif':
            #    if region[2] == 16 and region[1] in [3,40,20,23,26,30]:
            #        cv2.imwrite('region'+ j +'-'+ i +'.png', region[0])
            fft = np.fft.fft2(region[0])
            coords = max_coord(np.fft.fft2(region[0]))
            d = str(coords[3] if coords[3] < 3 else 0)
            if region[2] == 16 and region[1] == 30:
                x = remove_ring_points(fft, float(d), M=8, W=1)
                x = remove_dc(x,M=8)
                x = remove_percentile(x)
