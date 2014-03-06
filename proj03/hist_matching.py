import cv2
import numpy as np
import sys

L = 256

def calc_hist(src):
    hist_item = cv2.calcHist([src],[0],None,[256],[0,256])
    return hist_item

def eq_hist(src):
    x = np.zeros((L,1))

def uso():
    print 'Erro: nao foi possivel abrir a imagem fornecida'
    print 'Uso: python2 ' + sys.argv[0] + ' src dst'
    sys.exit()

def cdf(hist):
    cum_dist = np.zeros(hist.shape)
    cum_dist[0] = hist[0]
    for i in range(1,hist.shape[0]):
        cum_dist[i] = cum_dist[i-1][0] + hist[i][0]
    return cum_dist

def inverse(p):
    p *= (L-1)
    p = cv2.normalize(p,p,0,(L-1),cv2.NORM_MINMAX)
    G = np.int32(np.around(p))
    return G

def pass4(G):
    mapping = np.arange(L)
    for i in range(L):
        where = np.where(G == i)
        if len(where[0] > 0):
            mapping[i] = where[0][0]
    return mapping

if __name__ == '__main__':
    # Le imagem e converte para grayscale
    if len(sys.argv) != 3:
        uso()

    src = cv2.imread(sys.argv[1])
    if src == None:
        uso()
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    src = cv2.equalizeHist(src)

    hist = calc_hist(src)
    c = cdf(hist)
    inv = inverse(c)
    hist_matching = pass4(inv)

    dst = cv2.imread(sys.argv[2])
    dst = cv2.cvtColor(dst, cv2.cv.CV_BGR2GRAY)
    cv2.imwrite('dst_pre.jpg', dst)

    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst[i][j] = hist_matching[dst[i][j]]

    cv2.imwrite('src.jpg', src)
    cv2.imwrite('dst.jpg', dst)
