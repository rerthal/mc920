import cv2
import numpy as np
import sys

L = 256


def convolution(src, h, a=1, b=1):
    g = np.zeros(src.shape)
    f = src
    for x in range(a,src.shape[0]-a):
        for y in range(b,src.shape[1]-b):
            for i in range(-a,a+1):
                for j in range(-b,b+1):
                    g[x][y] += h[i+1][j+1] * f[x-i][y-j]
    return g

def thresholding(src):
    t = np.ones(src.shape) * (L-1)
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            if src[x][y] < (L/2):
                t[x][y] = 0
    return t

def uso():
    print 'Erro: nao foi possivel abrir a imagem fornecida'
    print 'Uso: python2 ' + sys.argv[0] + ' src dst'
    sys.exit()

def trim(x):
    y=np.delete(x, [0,x.shape[0]-1], 0)
    y=np.delete(y, [0,y.shape[1]-1], 1)
    return y

if __name__ == '__main__':
    # Le imagem e converte para grayscale
    if len(sys.argv) != 2:
        uso()

    src = cv2.imread(sys.argv[1])
    if src == None:
        uso()
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    cv2.imwrite('src.jpg', src)
    h = np.ones((3,3))
    h[0][0] = h[2][0] = h[0][2] = h[2][2] = 0.0
    h /= 5
    tst = np.ones((5,5))*10
    tst[2][2] = 90
    print convolution(tst, h)
    cv2.imwrite('dst1.jpg', trim(convolution(src,h)))

    h = np.ones((3,3))
    h[0][0] = h[2][0] = h[0][2] = h[2][2] = h[1][0] = 0.0
    h /= 4
    x = convolution(tst,h)
    y = trim(x)
    print convolution(tst,h)

    cv2.imwrite('dst2.jpg', trim(convolution(src,h)))

    src2 = cv2.imread('original2.jpg')
    src2 = cv2.cvtColor(src2, cv2.cv.CV_BGR2GRAY)
    cv2.imwrite('src_thr.jpg', src2)
    cv2.imwrite('dst_thr.jpg', thresholding(src2))
