import numpy as np
import cv2
import sys

def uso():
    print 'Erro: nao foi possivel abrir a imagem fornecida'
    print 'Uso: python2 ' + sys.argv[0] + ' src dst'
    sys.exit()

def salt_and_pepper(src, a, b, Pa=0, Pb=255):
    dst = np.copy(src)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] == a:
                dst[i][j] = Pa
            elif dst[i][j] == b:
                dst[i][j] = Pb
    return dst

def gaussian(src, a, b):
    dst = np.copy(src)
    noise = np.random.normal(a, b, dst.shape)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst[i][j] = dst[i][j] + noise[i][j]
    return dst

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

src = cv2.imread(sys.argv[1])
if src == None:
    uso()

src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
dst_sp = salt_and_pepper(src, 135, 17)
dst_ga = gaussian(src, 0.9, 1)

aniso = aniso_diff(src,gamma=0.25)
aniso_sp = aniso_diff(dst_sp,gamma=0.25)
aniso_ga = aniso_diff(dst_ga,gamma=0.25)

cv2.imwrite('src.jpg', src)
cv2.imwrite('dst_sp.jpg', dst_sp)
cv2.imwrite('dst_ga.jpg', dst_ga)
cv2.imwrite('aniso.jpg', aniso)
cv2.imwrite('aniso_sp.jpg', aniso_sp)
cv2.imwrite('aniso_ga.jpg', aniso_ga)
