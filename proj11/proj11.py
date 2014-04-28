import cv2
import sys
import numpy as np

if __name__ == '__main__':
    src = cv2.imread(sys.argv[1])
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(src, kernel, iterations = 1)
    dilation = cv2.dilate(src,kernel,iterations = 1)
    opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('src.jpg', src)
    cv2.imwrite('erosion.jpg', erosion)
    cv2.imwrite('dilation.jpg', dilation)
    cv2.imwrite('opening.jpg', opening)
    cv2.imwrite('closing.jpg', closing)

