import cv2
import numpy as np
import sys

def calc_hist(src, filename):
    # cria imagem com fundo branco
    h = np.ones((300,256,3)) * 255

    hist_item = cv2.calcHist([src],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    # converte histograma para inteiros
    hist=np.int32(np.around(hist_item))

    # desenha histograma (invertido)
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y), (0,0,0)) 

    # inverte a imagem
    h = np.flipud(h)
    cv2.imwrite(filename, h)

def uso():
    print 'Erro: nao foi possivel abrir a imagem fornecida'
    print 'Uso: python2 ' + sys.argv[0] + ' imagem'
    sys.exit()

if __name__ == '__main__':
    # Le imagem e converte para grayscale
    if len(sys.argv) == 1:
        uso()

    src = cv2.imread(sys.argv[1])
    if src == None:
        uso()
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)

    dst = cv2.equalizeHist(src)

    cv2.imwrite('src.jpg', src)
    cv2.imwrite('dst.jpg', dst)

    calc_hist(src, 'hist_src.jpg')
    calc_hist(dst, 'hist_dst.jpg')
