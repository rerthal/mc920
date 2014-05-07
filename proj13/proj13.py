import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def f(img, n, neighborhood=8):
    size = 2*n + 1
    if neighborhood == 4:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    erosion = cv2.erode(img, kernel, iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    return erosion - opening

def union(img, neighborhood):
    result = np.zeros(img.shape)
    for n in range(0, min(img.shape[0]/2, img.shape[1]/2) + 1):
        aux = f(img, n, neighborhood)
        result = result + aux
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j] = min(255, result[i][j])
    return result


if __name__ == '__main__':
#    for imgname in glob.glob('teste[12][ab].jpg'):
        imgname = 'img.png'
        img = cv2.imread(imgname, 0)
        bla = union(img, 8)
        cv2.imwrite(imgname[:-4] + '_8_final.jpg', bla)
        bla = union(img, 4)
        cv2.imwrite(imgname[:-4] + '_4_final.jpg', bla)
