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
import time
import scipy.stats

show_direc = True
show_watershed = True
show_markers = True
show_final = True
savefigs = True
print_direc_progress = False

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
    D = 16
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

def direc(image, w=4):
    img = np.copy(image)
    directional = np.zeros(img.shape)
    for i in range(w, img.shape[0]-w):
        if print_direc_progress:
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
            diff = abs(small[i][j] - large[i][j])
            if diff >= 45 and diff <= 135:
                rectified[i][j] = large[i][j]
                upd[i][j] = 1
    return rectified, upd

def crop(img, border):
    return img[border:img.shape[0]-border, border:img.shape[1]-border]

def mode(img, w=7):
    m = np.ones((img.shape))
    for i in range(w, img.shape[0]-w):
        if print_direc_progress:
            print 'moda [', 2*w+1, ']', i, '(', img.shape[0]-w, ')'
        for j in range(w, img.shape[1]-w):
            slice = img[i-w:i+w+1,j-w:j+w+1]
            m[i][j] = scipy.stats.mode(slice, axis=None)[0][0]
    return m

def orientation (img):
    directions = direc(img, w=4)
    small = mode(directions, w=7)
    large = mode(directions, w=22)
    small = crop(small,22)
    large = crop(large,22)
    rectified, X = rect(small, large)
   
    if show_direc:
        pylab.imshow(crop(directions,22))
        pylab.colorbar()
        pylab.savefig(basename + '_directions.jpg', bbox_inches='tight')

        pylab.imshow(small)
        pylab.savefig(basename + '_dir_small.jpg', bbox_inches='tight')

        pylab.imshow(large)
        pylab.savefig(basename + '_dir_large.jpg', bbox_inches='tight')

        pylab.imshow(rectified)
        pylab.savefig(basename + '_dir_rectified.jpg', bbox_inches='tight')

        changed = cv2.cvtColor(crop(np.copy(img),22), cv2.cv.CV_GRAY2BGR)
        changed[X == 1] = (255,0,0)
        pylab.figure()
        pylab.imshow(changed)
        pylab.savefig(basename + '_dir_rectified_pixels.jpg', bbox_inches='tight')

    return rectified, X


if __name__ == '__main__':
    directory = sys.argv[1]
    fingerprint = sys.argv[2]
    print fingerprint
    filename = "./fingerprints/" + directory + "/" + fingerprint
    basename = './fingerprints/' + directory + '/' + fingerprint.split('.')[0]
    if '.tif' not in fingerprint:
        print 'skipped ' + filename + ' (not a .tif file)'
        sys.exit()
    print filename
    start_timer = time.time()
    src = cv2.imread(filename)
    #src = src[250:350,150:250]
    #src = src[400:500,50:150]

    src_gray = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    src_gray_cropped = crop(src_gray, 22)
    cv2.imwrite(basename + '.jpg', src_gray_cropped)

    ## Secao 4.1
    # img_direc contem a informacao direcional em cada ponto
    # X contem os pixels em que a a informacao direcional calculada com janela pequena
    #   foi diferente daquela calculada com a janela grande
    img_direc, X = orientation(src_gray)


    ## Secao 4.2
    # water contem as linhas de watershed (segmentacao)
    _,water = pymorph.cwatershed(src_gray, pymorph.regmin(src_gray), return_lines= True)
    
    # remover elementos de conexidade 3 e 4 (yokoi 8-conexo)
    water_nc = nc_yokoi(water)

    water_lines = cv2.cvtColor(np.copy(src_gray), cv2.cv.CV_GRAY2BGR)
    water_lines[water == True] = (255,0,0)

    ## Secao 4.3
    # calcula a informacao direcional das linhas de watershed
    water_dir = direc(water, w=4)
    water_dir = mode(water_dir, w=7)
    water_dir = crop(water_dir, 22)
    water[water_nc == 3] = 0
    water[water_nc == 4] = 0

    if show_watershed:
        pylab.figure()
        pylab.imshow(crop(water_lines,22))
        pylab.savefig(basename + '_watershed.jpg', bbox_inches='tight')

        pylab.imshow(water_dir)
        pylab.colorbar()
        pylab.savefig(basename + '_water_dir.jpg', bbox_inches='tight')

    # marcar pixels em que a direcao no watershed eh perpendicular a imagem
    M = np.zeros(water_dir.shape)
    # marcadores -> 67.5 <= theta < 112.5
    M[abs(water_dir - img_direc) >= 67.5] = 1
    M[abs(water_dir - img_direc) > 112.5] = 0

    markers = cv2.cvtColor(np.copy(src_gray_cropped), cv2.cv.CV_GRAY2BGR)
    markers[M*crop(water,22)==1] = (255,0,0)

    if show_markers:
        f = pylab.figure()
        pylab.imshow(markers)
        pylab.savefig(basename + '_markers.jpg', bbox_inches='tight')

    ## Secao 4.4
    # Aplica transformada de distancia a X (secao 4.1)
    dist_X = mahotas.distance(X)
   
    ## Secao 4.5
    # aplicar abetura nos elementos de X centrados 
    reconnected = np.copy(src_gray)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if M[i][j] == 1 and crop(water,22)[i][j] == True:
                if X[i][j] == 1:
                    structuring_element = np.ones((1,dist_X[i][j] * 2 + 1), np.uint8)
                else:
                    structuring_element = np.ones((1,5), np.uint8)
                size = structuring_element.size
                #structuring_element = pymorph.serot(structuring_element, theta=img_direc[i][j])
                structuring_element = pymorph.serot(structuring_element, theta=90)
                window = cv2.morphologyEx(src_gray, cv2.MORPH_OPEN, structuring_element, iterations=10)[i+22-size/2:i+22+size/2+1,j]
                reconnected[i+22-size/2:i+22+size/2+1,j+22] = window

    cv2.imwrite(basename + '_reconnected.jpg', reconnected)
    print int(time.time() - start_timer), ' s'
