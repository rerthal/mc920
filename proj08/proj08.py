import numpy as np
import cv2
import sys
import math

L=256
k1 = 0.01
k2 = 0.03
k3 = k2/2
c1 = k1*L
c2 = k2*L
c3 = k3*L

def total_error(x, y):
    return sum(sum((x-y)**2))

def rms_error(x, y):
    return math.sqrt(total_error(x, y) / float(x.shape[0] * y.shape[1]))

def snr_error(x, y):
    denominator = sum(sum(y**2))
    return float(denominator) / total_error(x, y)

def ssim(x, y, alpha=1, beta=1, gamma=1, c1=c1, c2=c2, c3=c3):
    return pow(luminance(x, y, c1), alpha) * pow(contrast(x, y, c2), beta) * pow(structure(x, y, c3), gamma)

def mssim(x, y, window_size=11, alpha=1, beta=1, gamma=1, c1=c1, c2=c2, c3=c3):
    n = window_size
    M = 0.
    _mssim = 0.
    for i in range(n/2, x.shape[0]-n/2):
        for j in range(n/2, x.shape[1]-n/2):
            subx = x[i-n/2:i+n/2+1, j-n/2:j+n/2+1]
            suby = y[i-n/2:i+n/2+1, j-n/2:j+n/2+1]
            _mssim += ssim(subx, suby, alpha, beta, gamma, c1, c2, c3)
            M += 1
    return _mssim/M


def luminance(x, y, c=0):
    mu_x = x.mean()
    mu_y = y.mean()
    return (2 * mu_x * mu_y + c) / (pow(mu_x, 2) + pow(mu_y, 2) + c)

def contrast(x, y, c=0):
    sigma_x = x.std(ddof=1)
    sigma_y = y.std(ddof=1)
    return (2 * sigma_x * sigma_y + c) / (pow(sigma_x, 2) + pow(sigma_y, 2) + c)

def structure(x, y, c=0):
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std(ddof=1)
    sigma_y = y.std(ddof=1)
    sigma_xy = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sigma_xy += (x[i][j] - mu_x)*(y[i][j] - mu_y)
    sigma_xy /= (x.shape[0] * x.shape[1] - 1)
    return (sigma_xy + c) / (sigma_x * sigma_y + c)

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
    count = [0]*L
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            count[src[i][j]] += 1
    n = L/2
    a = count[:n].index(max(count[:n]))
    b = count[n:].index(max(count[n:]))
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

sp = salt_and_pepper_noise(src, 180, 122)
sp_median = cv2.medianBlur(sp, 3)
sp_gaussian = cv2.GaussianBlur(sp, ksize=(3,3), sigmaX=135)
sp_aniso = aniso_diff(sp)

ga = gaussian_noise(src, 2, 10)
ga_median = cv2.medianBlur(ga, 3)
ga_gaussian = cv2.GaussianBlur(ga, ksize=(3,3), sigmaX=135)
ga_aniso = aniso_diff(ga)


cv2.imwrite('src.jpg',src)
cv2.imwrite('sp.jpg',sp)
cv2.imwrite('sp_median.jpg',sp_median)
cv2.imwrite('sp_gaussian.jpg',sp_gaussian)
cv2.imwrite('sp_aniso.jpg',sp_aniso)
cv2.imwrite('ga.jpg',ga)
cv2.imwrite('ga_median.jpg',ga_median)
cv2.imwrite('ga_gaussian.jpg',ga_gaussian)
cv2.imwrite('ga_aniso.jpg',ga_aniso)


print mssim(src, ga_gaussian, 11)
print mssim(src, sp)
print mssim(src, sp_median)
print mssim(src, sp_gaussian)
print mssim(src, sp_aniso)
print mssim(src, ga)
print mssim(src, ga_median)
print mssim(src, ga_gaussian)
print mssim(src, ga_aniso)


print "Filtro gaussiano sobre ruido gaussiano         & %d & %.3f & %.3f & %.3f \\\\\hline" % (total_error(src, ga_gaussian), rms_error(src, ga_gaussian), snr_error(src, ga_gaussian), ssim(src, ga_gaussian, c1=c1, c2=c2, c3=c3))

print "Filtro gaussiano sobre ruido sal e pimenta     & %d & %.3f & %.3f & %.3f \\\\\hline" % (total_error(src, sp_gaussian), rms_error(src, sp_gaussian), snr_error(src, sp_gaussian), ssim(src, sp_gaussian, c1=c1, c2=c2, c3=c3))

print "Filtro da mediana sobre ruido gaussiano        & %d & %.3f & %.3f & %.3f \\\\\hline" % (total_error(src, ga_median), rms_error(src, ga_median), snr_error(src, ga_median), ssim(src, ga_median, c1=c1, c2=c2, c3=c3))

print "Filtro da mediana sobre ruido sal e pimenta    & %d & %.3f & %.3f & %.3f \\\\\hline" % (total_error(src, sp_median), rms_error(src, sp_median), snr_error(src, sp_median), ssim(src, sp_median, c1=c1, c2=c2, c3=c3))

print "Difusao anisotropica sobre ruido gaussiano     & %d & %.3f & %.3f & %.3f \\\\\hline" % (total_error(src, ga_aniso), rms_error(src, ga_aniso), snr_error(src, ga_aniso), ssim(src, ga_aniso, c1=c1, c2=c2, c3=c3))

print "Difusao anisotropica sobre ruido sal e pimenta & %d & %.3f & %.3f & %.3f \\\\\hline" % (total_error(src, sp_aniso), rms_error(src, sp_aniso), snr_error(src, sp_aniso), ssim(src, sp_aniso, c1=c1, c2=c2, c3=c3))


print "niter graph"
for x in range(1,500,50):
    filtered = aniso_diff(ga, niter=x)
    error    = ssim(src, filtered)
    print "(", x, ",", error, ")"

print "kappa graph"
for x in range(1,500,50):
    filtered = aniso_diff(ga, kappa=x)
    error    = ssim(src, filtered)
    print "(", x, ",", error, ")"

print "gamma graph"
for x in [z/500.0 for z in range(1, 500,50)]:
    filtered = aniso_diff(ga, gamma=x)
    error    = ssim(src, filtered)
    print "(", x, ",", error, ")"

