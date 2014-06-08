import numpy as np
import math
import cv2
import sys
from pymorph import erode
from pymorph import dilate
import os
import pylab
import mahotas
import scipy
from scipy import ndimage
import pymorph

def nc_yokoi(src, connectivity=8):
    def split(x):
        return x[1][2], x[0][2], x[0][1], x[0][0], x[1][0], x[2][0], x[2][1], x[2][2]
    def yokoi(x):
        x1, x2, x3, x4, x5, x6, x7, x8 = split(x)
        return  abs(x1 - x1*x2*x3) + \
                abs(x3 - x3*x4*x5) + \
                abs(x5 - x5*x6*x7) + \
                abs(x7 - x7*x8*x1)
    if connectivity == 4:
        img = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 1).astype(int)
    elif connectivity == 8:
        img = cv2.copyMakeBorder(1 - src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 1).astype(int)
    else:
        print 'Error. Connectivity = ', connectivity
        return
    ret = np.zeros(src.shape)
    for i in range(len(src)):
        for j in range(len(src[i])):
            ret[i][j] = yokoi(img[i:i+3,j:j+3])
    return ret


def direction(window, alpha=0.0):
    n = len(window)
    test_points = []
    for k in range(-n/2 + 1, n/2 + 1):
        x = int(round(n / 2 + k * math.cos(alpha)))
        y = int(round(n / 2 + k * math.sin(alpha)))
        test_points.append(window[x][y])
    return test_points

def dominant_direction(window, op):
#    D = 2*(window.shape[0] - 1)
    D = 8
    dominant_direction = None
    dominant_delta = 0.0
    for d in range(D):
        alpha = math.radians(d * 180.0 / float(D))
        beta = math.radians(90 + d * 180.0 / float(D))
        delta = op(direction(window, alpha)) - op(direction(window, beta))
        if delta > dominant_delta:
            dominant_delta = delta
            dominant_direction = d * 180.0 / float(D)
    return dominant_direction

def direc(image, w=7):
    img = np.copy(image)
    directional = np.zeros(img.shape)
    for i in range(w, img.shape[0]-w):
        print i, '(', img.shape[0]-w, ')'
        for j in range(w, img.shape[1]-w):
            slice = np.array(img)[i-w:i+w+1, j-w:j+w+1]
            directional[i][j] = None
            alpha = dominant_direction(slice, np.std)
            if alpha != None:
                directional[i][j] = alpha
    return directional

def rect(small, large):
    rectified = np.copy(small)
    upd = np.zeros(small.shape)
    for i in range(small.shape[0]):
        for j in range(small.shape[1]):
            if abs(small[i][j] - large[i][j]) == 90:
                rectified[i][j] = large[i][j]
                upd[i][j] = 1
    return rectified, upd

def crop(img, border):
    return img[border:img.shape[0]-border, border:img.shape[1]-border]

def orientation (img):
    small = direc(img, w=7)
    small = crop(small, 22)
    large = direc(img, w=22)
    large = crop(large, 22)
    rectified, X = rect(small, large)
    return rectified, X


if __name__ == '__main__':
        pylab.gray()
#    for directory in os.listdir("./fingerprints/"):
        directory = "2002Db2a"
        print directory
#        for fingerprint in os.listdir("./fingerprints/" + directory):
        for fingerprint in ["1_5.tif"]:
#        for fingerprint in os.listdir("./fingerprints/" + directory):
            filename = "./fingerprints/" + directory + "/" + fingerprint
            src = cv2.imread(filename)
           # src = src[200:400,100:300]

            src_gray = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
            src_binary = cv2.threshold(src_gray, 80, 255, cv2.THRESH_BINARY)[1] / 255

            ## Secao 4.1
            # img_direc contem a informacao direcional em cada ponto
            # X contem os pixels em que a a informacao direcional calculada com janela pequena
            #   foi diferente daquela calculada com a janela grande
            img_direc, X = orientation(src_gray)

            #pylab.imshow(src)
            #pylab.show()
            #pylab.imshow(img_direc)
            #pylab.show()
            #pylab.imshow(X)
            #pylab.show()

            ## Secao 4.2
            # water contem as linhas de watershed (segmentacao)
            _,water = pymorph.cwatershed(src_gray, pymorph.regmin(src_gray), return_lines= True)
            #pylab.imshow(water)
            #pylab.show()

            ## Secao 4.3
            # calcula a informacao direcional das linhas de watershed
            water_dir = direc(water, w=7)
            water_dir = crop(water_dir, 22)
            
            # remover elementos de conexidade 3 e 4 (yokoi 8-conexo)
            water_nc = nc_yokoi(water)
            water[water_nc == 3] = 0
            water[water_nc == 4] = 0

            # marcar pixels em que a direcao no watershed eh perpendicular a imagem
            M = np.zeros(water_dir.shape)
            M[abs(water_dir - img_direc) == 90] = 1
            #pylab.imshow(M)
            #pylab.show()

            ## Secao 4.4
            # Aplica transformada de distancia a X (secao 4.1)
            dist_X = mahotas.distance(X)
           
            ## Secao 4.5
            # aplicar abetura nos elementos de X centrados 
            answer = crop(np.copy(src_gray),22)
            for i in range(len(X)):
                for j in range(len(X[i])):
                    if M[i][j] == 1:
                        if X[i][j] == 1:
                            structuring_element = np.ones((1,dist_X[i][j] * 2 + 1), np.uint8)
                        else:
                            structuring_element = np.ones((1,5), np.uint8)
                        structuring_element = pymorph.serot(structuring_element, theta=img_direc[i][j])
                        pixel = cv2.morphologyEx(src_gray, cv2.MORPH_OPEN, structuring_element)[i][j]
                        answer[i][j] = pixel
            pylab.imshow(answer)
            pylab.show()

           # background = b(src)
           # mask = field_detection_block(src - background)
           # filtered = filter(src, mask)
           # cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_src.jpg', src)
           # cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_background.jpg', background)
           # cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_mask.jpg', mask)
           # cv2.imwrite("./fingerprints/" + directory + "/" + fingerprint.split('.')[0] + '_filtered.jpg', filtered)
            print fingerprint
