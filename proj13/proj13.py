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
    new_img = cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_CONSTANT, value=0)
    erosion = cv2.erode(new_img, kernel, iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    erosion = erosion[n:len(erosion)-n, n:len(erosion[0])-n]
    opening = opening[n:len(opening)-n, n:len(opening[0])-n]
    return erosion - opening

def union(img, neighborhood):
    result = np.zeros(img.shape)
    for n in range(0, min(img.shape[0]/2, img.shape[1]/2) + 1):
        aux = f(img, n, neighborhood)
        result = result + aux
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j] = min(1, result[i][j])
    return result * 255


if __name__ == '__main__':
    for imgname in glob.glob('teste[012][ab].png'):
        print imgname
        img = cv2.imread(imgname, 0)
        img /= 255
        bla = union(img, 8)
        cv2.imwrite(imgname[:-4] + '_8_final.png', bla)
        bla = union(img, 4)
        cv2.imwrite(imgname[:-4] + '_4_final.png', bla)
