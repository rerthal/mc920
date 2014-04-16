import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

L=256

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

def max_coord(f,M=8):
    # shift zero-frequency term to the centre of the array
    fourier = np.fft.fftshift(f)
    centre=(M,M)
    max = (0,0,0,0)
    for i in range(1,len(fourier)):
        for j in range(1,len(fourier[i])):
            norm = intensity(fourier[i][j])
            if norm > max[2] and (i,j) != centre:
                dist = math.sqrt((i-centre[0])**2 + (j-centre[1])**2)
                max = (i-centre[0],j-centre[0],norm,dist)
    return max

def remove_ring_points(fourier, r, M=8, W=1):
    centre = (M,M)
    circle = np.ones((2*M, 2*M))
    cv2.circle(circle, centre, int(round(r)), color=0)
    return fourier*circle

def remove_dc(fourier, M=8):
    centre = (M,M)
    aux = np.ones((2*M, 2*M))
    aux[(centre)] = 0
    return fourier*aux

def remove_percentile(fourier,t=0.05):
    M = fourier.shape[0]/2
    f = intensity_vec(fourier)
    f_max = np.amax(f)
    if f_max > 0:
        f *= (255/f_max)
    h = cv2.calcHist([f.astype(np.uint8)],[0],None,[256],[0,256])
    accum = 0
    percentile = np.ones((2*M, 2*M))
    for i in range(L-1,-1,-1):
        accum += h[i]
        if accum > ((2*M)**2)*t:
            break
    for i in range(len(fourier)):
        for j in range(len(fourier[i])):
            if f[i][j] >= accum:
                percentile[i][j] = 0
    return percentile*fourier

def get_noise_frequency_level(fourier):
    return np.amax(intensity_vec(fourier))

def HMb(fourier, npb):
    M = fourier.shape[0]/2
    h = np.zeros((2*M,2*M), dtype=complex)
    f = intensity_vec(fourier)
    for i in range(len(fourier)):
        for j in range(len(fourier[i])):
            if f[i][j] >= npb:
                h[i][j] = fourier[i][j]
    return h

def band_pass_filter(fourier, f, W=1):
    return fourier

def gaussian_directional_filter(theta, size=5, sigma=1):
    mask = np.ones((size,size))
    return mask

if __name__ == '__main__':
    fingerprints = glob.glob("Fingerprints/*.tif")
    for fingerprint in fingerprints:
        print fingerprint
        image = cv2.imread(fingerprint)
        image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        lines = {}
        for region in slice(image):
            if len(region[0]) != 16:
                continue
            i, j = region[1:3]
            i_str = str(i)
            j_str = str(j)
            if len(i_str) == 1:
                i_str = '0' + i_str
            if len(j_str) == 1:
                j_str = '0' + j_str
            fft = np.fft.fft2(region[0])
            coords = max_coord(np.fft.fft2(region[0]))
            d = str(coords[3] if coords[3] <= 3 else 0)
            if j%2 == 0 and i%2 == 0:
                fft = np.fft.fftshift(fft)
                f = remove_ring_points(fft, float(d), M=8, W=1)
                f = remove_dc(f,M=8)
                f = remove_percentile(f)
                npb = get_noise_frequency_level(f)
                fft = np.fft.fftshift(fft)
                h = HMb(fft, npb)
                h = np.fft.ifft2(h)
                h = intensity_vec(h)
                h *= 255/(np.amax(h))
                h = h.astype(np.uint8)
                # first column
                if i == 2:
                    lines[j] = h
                else:
                    lines[j] = np.concatenate((lines[j], h), axis=1)
        for key in sorted(lines.keys()):
            if key == 2:
                img = lines[key]
            else:
                img = np.concatenate((img,lines[key]))
        cv2.imwrite(fingerprint[:-4] + '_intermediate.jpg', img)
