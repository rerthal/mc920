import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

L=256

def convolution(src, h, a=1, b=1):
    return cv2.filter2D(src, -1, h)

def truncate(src):
    for i in range(len(src)):
        for j in range(len(src[i])):
            src[i][j] = min(src[i][j], 255)
            src[i][j] = max(src[i][j], 0)
    return src

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

# Retorna o quadrado da distancia entre p1 e p2
def dist2_pt(p1, p2):
    return float((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def band_pass_filter(fourier, f):
    M = fourier.shape[0]/2
    BW = fourier.shape[0]/16
    centre = (M, M)
    result = np.ones(fourier.shape)
    f = int(float(f))
    if f == 0:
        return result
    for i in range(len(fourier)):
        for j in range(len(fourier[i])):
            d = dist2_pt((i,j), centre)
            Hlp = math.exp((-1/2.0)*(d)/((f+BW)**2))
            Hhp = 1 - Hlp
            result[i][j] = Hlp * Hhp
    return result

mask0O0 = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
mask022 = [[0, 0, 0, 0, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]]
mask045 = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
mask067 = [[0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0]]
mask090 = [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]
mask112 = [[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1]]
mask135 = [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]
mask157 = [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1]]

def gaussian_directional_filter(theta, size=5, sigma=1):
    theta = (theta+180)/2
    mask = np.ones((size,size))
    for i in range(-size/2 + 1, size/2 + 1):
        for j in range(-size/2 + 1, size/2 + 1):
            mask[i][j] = math.exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * math.pi * sigma * sigma)
    if theta < 15:
        return mask * mask0O0
    elif theta < 37:
        return mask * mask022
    elif theta < 59:
        return mask * mask045
    elif theta < 81:
        return mask * mask067
    elif theta < 103:
        return mask * mask090
    elif theta < 125:
        return mask * mask112
    elif theta < 147:
        return mask * mask135
    else:
        return mask * mask157

if __name__ == '__main__':
    fingerprints = glob.glob("Fingerprints/*.tif")
    for fingerprint in fingerprints:
        print fingerprint
        image = cv2.imread(fingerprint)
        image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        lines_intermediate = {}
        lines_final = {}
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
            if coords[0] != 0:
                theta = math.degrees(math.atan(coords[1]/coords[0]))
            else:
                theta = 180
            if j%2 == 0 and i%2 == 0:
                fft = np.fft.fftshift(fft)
                f = remove_ring_points(fft, float(d), M=8, W=1)
                f = remove_dc(f,M=8)
                f = remove_percentile(f)
                npb = get_noise_frequency_level(f)
                fft = np.fft.fftshift(fft)
                # Fase 1
                hmb = HMb(fft, npb)
                hmb = np.fft.ifft2(hmb)
                hmb_intensity = intensity_vec(hmb)
                hmb_intensity *= 255/(np.amax(hmb_intensity))
                hmb_intensity = hmb_intensity.astype(np.uint8)
                # Fase 2
                fft = np.fft.fftshift(fft)
                hbb = band_pass_filter(fft, d)
                gtheta = gaussian_directional_filter(theta)
                #hghb = convolution(np.fft.fft2(hmb)* hbb, gtheta)
                hghb = np.fft.fft2(hmb)*hbb
                hghb = np.fft.ifft2(hghb)
                hghb_intensity = intensity_vec(hghb)
                hghb_intensity = convolution(hghb_intensity, gtheta)
                hghb_intensity *= 255/(np.amax(hghb_intensity))
                #hghb_intensity = truncate(hghb_intensity)
                hghb_intensity = hghb_intensity.astype(np.uint8)
                '''
                if hbb[0][0] != 1 and i==14 and j==10:
                    print 'hmb'
                    plt.imshow(intensity_vec(hmb))
                    plt.show()
                    print 'hghb'
                    plt.imshow(intensity_vec(hghb))
                    plt.show()
                '''
                # first column
                if i == 2:
                    lines_intermediate[j] = hmb_intensity
                    lines_final[j] = hghb_intensity
                else:
                    lines_intermediate[j] = np.concatenate((lines_intermediate[j], hmb_intensity), axis=1)
                    lines_final[j] = np.concatenate((lines_final[j], hghb_intensity), axis=1)
        for key in sorted(lines_intermediate.keys()):
            if key == 2:
                img_intermediate = lines_intermediate[key]
                img_final = lines_final[key]
            else:
                img_intermediate = np.concatenate((img_intermediate,lines_intermediate[key]))
                img_final = np.concatenate((img_final,lines_final[key]))
        cv2.imwrite(fingerprint[:-4] + '_intermediate.jpg', img_intermediate)
        cv2.imwrite(fingerprint[:-4] + '_final.jpg', img_final)
