import glob
import numpy as np
import cv2
import math

import matplotlib.pyplot as plt

def even_slices(image, M=8):
    for i in range(1, image.shape[0] / M):
        for j in range(1, image.shape[1] / M):
            if i % 2 == 0 and j % 2 == 0:
                sub_image = np.array(image)[(i-1)*M: (i+1)*M, (j-1)*M: (j+1)*M]
                if len(sub_image) > 0: yield (np.asarray(sub_image), i, j)

def max_frequency(image,M=8):
    centre=(M,M)
    max = (0,0,0,0)
    for i in range(1,len(image)):
        for j in range(1,len(image[i])):
            norm = abs(image[i][j])**2
            if norm > max[2] and (i,j) != centre:
                dist = math.sqrt((i-centre[0])**2 + (j-centre[1])**2)
                max = (i-centre[0],j-centre[1],norm,dist)
    return int(math.ceil(max[3]))

def Npb(image):
    circle = np.ones(image.shape)
    cv2.circle(circle, (image.shape[0]/2, image.shape[1]/2), max_frequency(image), color=0)
    removed_ring = image*circle
    removed_ring[(image.shape[0] / 2, image.shape[1] / 2)] = 0
    percentile = np.percentile(removed_ring, 5)
    removed_percentile = np.vectorize(lambda x: 0 if x >= percentile else abs(x))(removed_ring)
    return np.amax(removed_percentile)

def HMb(region, M=8):
    npb = Npb(region)
    hmb = np.zeros((2*M,2*M), dtype=complex)
    for i in range(len(region)):
        for j in range(len(region[i])):
            if abs(region[i][j]) >= npb: hmb[i][j] = region[i][j]
    return hmb

def Hghb(region, M=8):
    result = np.ones((2*M,2*M), dtype=complex)
    for i in range(len(region)):
        for j in range(len(region[i])):
            d = np.linalg.norm([i - region.shape[0]/2, j - region.shape[1]/2])
            Hlp = math.exp((-1/2.0)*(d**2)/((max_frequency(region)+region.shape[0]/16)**2))
            Hhp = 1 - math.exp((-1/2.0)*(d**2)/((max_frequency(region))**2))
            result[i][j] = Hlp * Hhp
    return result * HMb(region)

if __name__ == '__main__':
    #for fingerprint in glob.glob("Fingerprints/*.tif"):
        fingerprint = "Fingerprints/1_1.tif"
        image = cv2.cvtColor(cv2.imread(fingerprint), cv2.cv.CV_BGR2GRAY)
        intermediate_image = np.zeros(image.shape)
        final_image = np.zeros(image.shape)
        for region, ki, kj in even_slices(image):
            fft = np.fft.fft2(region)
            fft_shift = np.fft.fftshift(fft)
            hmb = np.fft.ifft2(np.fft.ifftshift(HMb(fft_shift)))
            hghb = np.fft.ifft2(np.fft.ifftshift(Hghb(fft_shift)))
            hmb_intensity = (hmb * 255) / (np.amax(hmb))
            for i in range(16):
                for j in range(16):
                    intermediate_image[(ki) * 8 + i][(kj) * 8 + j] = abs(hmb_intensity[i][j])
                    final_image[(ki) * 8 + i][(kj) * 8 + j] = hghb[i][j]
        cv2.imwrite(fingerprint[:-4] + '_intermediate.jpg', intermediate_image)
        cv2.imwrite(fingerprint[:-4] + '_final.jpg', final_image)
        print fingerprint[:-4]