import glob
import numpy as np
import cv2
import math

def even_slices(image, M=8):
    for i in range(1, image.shape[0] / M):
        for j in range(1, image.shape[1] / M):
            if i % 2 == 0 and j % 2 == 0:
                sub_image = [line[(i-1)*M: (i+1)*M] for line in image[(j-1)*M: (j+1)*M]]
                if len(sub_image) > 0: yield (np.asarray(sub_image), i, j)

def max_frequency(image,M=8):
    max = (0,0,0)
    for i in range(1, len(image)):
        for j in range(1, len(image[i])):
            if abs(image[i][j]) > max[2]:
                if(i,j) != (M,M): max = (i-M, j-M, abs(image[i][j]))
    return int(np.linalg.norm([max[0] - M, max[1] - M]))

def Npb(image):
    circle = np.ones(image.shape)
    cv2.circle(circle, image.shape, max_frequency(image), color=0)
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
    return np.fft.ifft2(hmb)

def Hghb(region, M=8):
    result = np.ones((2*M,2*M), dtype=complex)
    for i in range(len(region)):
        for j in range(len(region[i])):
            d = np.linalg.norm([i - image.shape[0]/2, j - image.shape[1]/2])
            Hlp = math.exp((-1/2.0)*(d)/((max_frequency(image)+image.shape[0]/16)**2))
            Hhp = 1 - math.exp((-1/2.0)*(d)/((max_frequency(image))**2))
            result[i][j] = Hlp * Hhp
    return np.fft.ifft2(result)

if __name__ == '__main__':
    for fingerprint in glob.glob("Fingerprints/*.tif"):
        image = cv2.cvtColor(cv2.imread(fingerprint), cv2.cv.CV_BGR2GRAY)
        intermediate_image = np.zeros((400,400))
        final_image = np.zeros((400,400))
        for region, ki, kj in even_slices(image):
            fft = np.fft.fft2(region)
            fft_shift = np.fft.fftshift(fft)
            hmb = HMb(fft_shift)
            hghb = Hghb(fft_shift)
            hmb_intensity = (hmb * 255) / (np.amax(hmb))
            hghb_intensity = (hghb * 255) / (np.amax(hghb))
            for j in range(16):
                for i in range(16):
                    intermediate_image[(kj) * 8 + j][(ki) * 8 + i] = abs(hmb_intensity[j][i])
                    final_image[(kj) * 8 + j][(ki) * 8 + i] = abs(hghb_intensity[j][i])
        cv2.imwrite(fingerprint[:-4] + '_intermediate.jpg', intermediate_image)
        cv2.imwrite(fingerprint[:-4] + '_final.jpg', final_image)
        print fingerprint[:-4]