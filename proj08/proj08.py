import numpy as np
import cv2
import sys

def total_error(src, filtered):
    error = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            print error
            error = error + (src[i][j] - filtered[i][j]) * (src[i][j] - filtered[i][j])
    return error

def rms_error(src, filtered):
    return np.sqrt(total_error(src, filtered) / float(src.shape[0] * src.shape[1]))

def snr_error(src, filtered):
    sum = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            sum = sum + (filtered * filtered)
    return float(sum) / total_error(src, filtered)

def ssim_error(src, filtered):
    return 0

def aniso_diff(img,niter=10,kappa=50,gamma=0.1,step=(1.,1.)):
    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in xrange(niter):
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
        gE = np.exp(-(deltaE/kappa)**2.)/step[1]

        S = gS*deltaS
        E = gE*deltaE

        NS[:] = S
        EW[:] = E

        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        imgout += gamma*(NS+EW)

    return imgout

def salt_and_pepper_noise(src, a, b, Pa=0, Pb=255):
    dst = np.copy(src)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] == a:
                dst[i][j] = Pa
            elif dst[i][j] == b:
                dst[i][j] = Pb
    return dst

def gaussian_noise(src, a, b):
    dst = np.copy(src)
    noise = np.random.normal(a, b, dst.shape)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst[i][j] = dst[i][j] + noise[i][j]
    return dst

src = cv2.imread(sys.argv[1])
src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)

image_sp = salt_and_pepper_noise(src, 135, 17)
image_sp_noise_median_filter = cv2.medianBlur(image_sp, 3)
image_sp_noise_gaussian_filter = cv2.GaussianBlur(image_sp, ksize=(3,3), sigma1=135)
image_sp_noise_aniso_diff = aniso_diff(image_sp)

image_ga = gaussian_noise(src, 2, 10)
image_ga_noise_median_filter = cv2.medianBlur(image_ga, 3)
image_ga_noise_gaussian_filter = cv2.GaussianBlur(image_ga, ksize=(3,3), sigma1=135)
image_ga_noise_aniso_diff = aniso_diff(image_ga)

print "Filtro gaussiano sobre ruido gaussiano         &", total_error(src, image_ga_noise_gaussian_filter), "&", rms_error(src, image_ga_noise_gaussian_filter), "&", snr_error(src, image_ga_noise_gaussian_filter), "&", ssim_error(src, image_ga_noise_gaussian_filter),  "\\ \hline"
print "Filtro gaussiano sobre ruido sal e pimenta     &", total_error(src, image_sp_noise_gaussian_filter), "&", rms_error(src, image_sp_noise_gaussian_filter), "&", snr_error(src, image_sp_noise_gaussian_filter), "&", ssim_error(src, image_sp_noise_gaussian_filter),  "\\ \hline"
print "Filtro da mediana sobre ruido gaussiano        &", total_error(src, image_ga_noise_median_filter), "&", rms_error(src, image_ga_noise_median_filter), "&", snr_error(src, image_ga_noise_median_filter), "&", ssim_error(src, image_ga_noise_median_filter),  "\\ \hline"
print "Filtro da mediana sobre ruido sal e pimenta    &", total_error(src, image_sp_noise_median_filter), "&", rms_error(src, image_sp_noise_median_filter), "&", snr_error(src, image_sp_noise_median_filter), "&", ssim_error(src, image_sp_noise_median_filter),  "\\ \hline"
print "Difusao anisotropica sobre ruido gaussiano     &", total_error(src, image_ga_noise_aniso_diff), "&", rms_error(src, image_ga_noise_aniso_diff), "&", snr_error(src, image_ga_noise_aniso_diff), "&", ssim_error(src, image_ga_noise_aniso_diff),  "\\ \hline"
print "Difusao anisotropica sobre ruido sal e pimenta &", total_error(src, image_sp_noise_aniso_diff), "&", rms_error(src, image_sp_noise_aniso_diff), "&", snr_error(src, image_sp_noise_aniso_diff), "&", ssim_error(src, image_sp_noise_aniso_diff),  "\\ \hline"

print "niter graph"
for x in range(1,500):
    filtered = aniso_diff(image_ga, niter=x)
    error    = ssim_error(src, filtered)
    print "(", x, ",", error, ")"

print "kappa graph"
for x in range(1,500):
    filtered = aniso_diff(image_ga, kappa=x)
    error    = ssim_error(src, filtered)
    print "(", x, ",", error, ")"

print "gamma graph"
for x in [z/500.0 for z in range(1, 500)]:
    filtered = aniso_diff(image_ga, gamma=x)
    error    = ssim_error(src, filtered)
    print "(", x, ",", error, ")"
