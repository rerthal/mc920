import numpy as np
import cv2
import pylab
import mahotas

def print_img(img):
    print '====='
    for line in img:
        for column in line:
            print column.astype(int),
        print
    print '====='

def split(x):
    return x[1][2], x[0][2], x[0][1], x[0][0], x[1][0], x[2][0], x[2][1], x[2][2]

def rutovitz(x):
    return (abs(x[1][2] - x[0][2]) + \
            abs(x[0][2] - x[0][1]) + \
            abs(x[0][1] - x[0][0]) + \
            abs(x[0][0] - x[1][0]) + \
            abs(x[1][0] - x[2][0]) + \
            abs(x[2][0] - x[2][1]) + \
            abs(x[2][1] - x[2][2]) + \
            abs(x[2][2] - x[1][2])) / 2


def yokoi(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = split(x)
    return  abs(x1 - x1*x2*x3) + \
            abs(x3 - x3*x4*x5) + \
            abs(x5 - x5*x6*x7) + \
            abs(x7 - x7*x8*x1)

def nc_rutovitz(src):
    img = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 1).astype(int)
    ret = np.zeros(src.shape)
    for i in range(len(src)):
        for j in range(len(src[i])):
            ret[i][j] = rutovitz(img[i:i+3,j:j+3])
    return ret[1:-1,1:-1]

def nc_yokoi(src, connectivity=4):
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


if __name__ == '__main__':
    pylab.axis('off')

    img = cv2.imread('img2.png')
    img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    img /= 255

    # Rutovitz
    rut = nc_rutovitz(img)
    pylab.imshow(rut, interpolation='nearest')
    pylab.savefig('rutovitz.png', bbox_inches='tight')

    # Yokoi 4
    yokoi4 = nc_yokoi(img, 4)
    pylab.imshow(yokoi4, interpolation='nearest')
    pylab.savefig('yokoi4.png', bbox_inches='tight')

    # Yokoi 8
    yokoi8 = nc_yokoi(img, 8)
    pylab.imshow(yokoi8, interpolation='nearest')
    pylab.savefig('yokoi8.png', bbox_inches='tight')

    # Transformada de distancia
    img = cv2.imread('img.png')
    img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    dist = mahotas.distance(img, metric='euclidean')
    pylab.imshow(dist, interpolation='nearest')
    pylab.gray()
    pylab.savefig('dist.png', bbox_inches='tight')
