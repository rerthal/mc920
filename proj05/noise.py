import cv2
import numpy as np
import sys

L = 256

def salt_and_pepper(src, a, b, Pa=0, Pb=255):
    dst = np.copy(src)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] == a:
                dst[i][j] = Pa
            elif dst[i][j] == b:
                dst[i][j] = Pb
    return dst

def uso():
    print 'Erro: nao foi possivel abrir a imagem fornecida'
    print 'Uso: python2 ' + sys.argv[0] + ' src dst'
    sys.exit()

if __name__ == '__main__':
    # Le imagem e converte para grayscale
    if len(sys.argv) != 2:
        uso()

    src = cv2.imread(sys.argv[1])
    if src == None:
        uso()
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    dst_sp = salt_and_pepper(src, 135, 17)
    dst_sp_avg_blur3 = cv2.blur(dst_sp, (3, 3))
    dst_sp_avg_blur5 = cv2.blur(dst_sp, (5, 5))
    dst_sp_median_blur = cv2.medianBlur(dst_sp, 3)
    cv2.imwrite('src.jpg', src)
    cv2.imwrite('dst_sp.jpg', dst_sp)
    cv2.imwrite('dst_sp_avg3.jpg', dst_sp_avg_blur3)
    cv2.imwrite('dst_sp_avg5.jpg', dst_sp_avg_blur5)
    cv2.imwrite('dst_sp_median.jpg', dst_sp_median_blur)

