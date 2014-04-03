import cv2
import sys

def slice(image, W=8):
    lines   = image.shape[0] / W
    columns = image.shape[1] / W
    for i in range(lines):
        for j in range(columns):
            yield image[j*W: (j+1)*W][i*W: (i+1)*W]

src = cv2.imread(sys.argv[1])
src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)

slice(src)
#for i in slice(src):
#    print i.shape