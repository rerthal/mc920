import cv2
import numpy as np
import sys
import math

L = 256 

def uso():
    print 'Erro: nao foi possivel abrir a imagem fornecida'
    print 'Uso: python2 ' + sys.argv[0] + ' src dst'
    sys.exit()

def combine(gx, gy):
    return (gx+gy)/2


def thresholding(src, thr=L/2):
    t = np.ones(src.shape) * (L-1)
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            if src[x][y] < thr:
                t[x][y] = 0
    return t

if __name__ == '__main__':
    # Le imagem e converte para grayscale
    if len(sys.argv) != 2:
        uso()

    src = cv2.imread(sys.argv[1])
    if src == None:
        uso()
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)

    # Sobel
    kernel_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel_sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel_x = cv2.filter2D(src, -1, kernel_sobel_x)
    sobel_y = cv2.filter2D(src, -1, kernel_sobel_y)
    sobel = combine(sobel_x, sobel_y)

    # Prewitt
    kernel_prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1]])
    kernel_prewitt_y = np.array([[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1]])
    prewitt_x = cv2.filter2D(src, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(src, -1, kernel_prewitt_y)
    prewitt = combine(prewitt_x, prewitt_y)

    # Laplacian
    kernel_laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian = cv2.filter2D(src, -1, kernel_laplacian)

    # Roberts
    kernel_roberts_x = np.array([[1,0],[0,-1]])
    kernel_roberts_y = np.array([[0,1],[-1,0]])
    roberts_x = cv2.filter2D(src, -1, kernel_roberts_x)
    roberts_y = cv2.filter2D(src, -1, kernel_roberts_y)
    roberts = combine(roberts_x, roberts_y)

    # Escreve as imagens
    cv2.imwrite('src.jpg', src)
    cv2.imwrite('sobel.jpg', sobel)
    cv2.imwrite('sobel_x.jpg', sobel_x)
    cv2.imwrite('sobel_y.jpg', sobel_y)
    cv2.imwrite('prewitt_x.jpg', prewitt_x)
    cv2.imwrite('prewitt_y.jpg', prewitt_y)
    cv2.imwrite('prewitt.jpg', prewitt)
    cv2.imwrite('laplacian.jpg', laplacian)
    cv2.imwrite('roberts_x.jpg', roberts_x)
    cv2.imwrite('roberts_y.jpg', roberts_y)
    cv2.imwrite('roberts.jpg', roberts)
