import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def slice(image, W=8):
    lines   = image.shape[0] / W
    columns = image.shape[1] / W
    for i in range(1,lines):
        for j in range(1,columns):
            sub_image = [line[(i-1)*W: (i+1)*W] for line in image[(j-1)*W: (j+1)*W]]
            if len(sub_image) > 0:
                yield (np.asarray(sub_image), i, j)

def max_coord(fourier,W=8):
    # shift zero-frequency term to the centre of the array
    fourier = np.fft.fftshift(fourier)
    centre=(W,W)
    max = (0,0,0,0)
    for i in range(1,len(fourier)):
        for j in range(1,len(fourier[i])):
            norm = (fourier[i][j].real * fourier[i][j].real) + (fourier[i][j].imag * fourier[i][j].imag)
            if norm > max[2] and (i,j) != centre:
                dist = math.sqrt((i-centre[0])**2 + (j-centre[1])**2)
                max = (i-centre[0],j-centre[0],norm,dist)
    return max


if __name__ == '__main__':
    fingerprints = glob.glob("Fingerprints/1_*.tif")
    for fingerprint in fingerprints:
        image = cv2.imread(fingerprint)
        image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        # We'll print frequency for all images
        i = fingerprint[-5] # get image suffix
        f_all = open('all' + i +'.txt', 'w')
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
            if fingerprint == 'Fingerprints/1_1.tif':
                if region[2] == 16 and region[1] in [3,40,20,23,26,30]:
                    cv2.imwrite('region'+ j +'-'+ i +'.png', region[0])
            fft = np.fft.fft2(region[0])
            coords = max_coord(np.fft.fft2(region[0]))
            d = str(coords[3] if coords[3] < 3 else 0)
            f_all.write(i + ' ' + j + ' ' + d + '\n')
            if region[2] == 16 and fingerprint == 'Fingerprints/1_1.tif':
                f_line.write(i + ' ' + d +'\n')
                if region[1] in [3,40,20,23,26,30]:
                    plt.imshow(np.log(np.abs(np.fft.fftshift(fft))**2))
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.savefig('fft' + i +'.png', bbox_inches='tight', pad_inches=0)
