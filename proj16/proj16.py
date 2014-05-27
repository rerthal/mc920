import numpy as np
import cv2
import sys
import mahotas
import scipy
from scipy import ndimage
import pylab
import pymorph

def print_img(img):
    print '====='
    for line in img:
        for column in line:
            print column,
        print
    print '====='

def geodesic_dilation(img, mask, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))):
    new_img = np.copy(img)
    while True:
        print 1
        prev_img = np.copy(new_img)
        new_img = cv2.dilate(new_img, kernel, iterations=1)
        new_img = cv2.bitwise_and(new_img, mask)
        changed = False
        for i in range(len(prev_img)):
            for j in range(len(prev_img[i])):
                if prev_img[i][j] != new_img[i][j]:
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return new_img

def geodesic_reconstruction_1d(f, g):
    new = np.copy(f)
    while True:
        old = np.copy(new)
        print 'old', old
        new = mahotas.dilate(g, np.asarray([1]))
        print 'new', new
        changed = False
        for i in range(len(g)):
            new[i] = min(new[i], f[i])
            if new[i] != old[i]:
                changed = True
        print 'newnew', new
        if not changed:
            break
    print '---'
    print new

# based on http://pythonvision.org/basic-tutorial
def watershed(img):
    # Diminui ruidos
    imgf = ndimage.gaussian_filter(img, 16)
    mahotas.imsave('dnaf.jpeg', imgf)
    rmax = pymorph.regmax(imgf)
    
    T = mahotas.thresholding.otsu(imgf)
    dist = ndimage.distance_transform_edt(imgf > T)
    dist = dist.max() - dist
    dist -= dist.min()
    dist = dist/float(dist.ptp()) * 255
    dist = dist.astype(np.uint8)
    mahotas.imsave('dist.jpeg', dist)

    seeds,nr_nuclei = ndimage.label(rmax)
    nuclei = pymorph.cwatershed(dist, seeds)
    mahotas.imsave('watershed.jpeg', nuclei)
    #pylab.imshow(nuclei)
    #pylab.show()


# Dilatacao geodesica
img = cv2.imread('image.png')
img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
img /= 255

mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.cv.CV_BGR2GRAY)
mask /= 255

print_img(img)
print_img(mask)
new_img = geodesic_dilation(img, mask) * 255
cv2.imwrite('./geodesic_dilation.png', new_img)

# Reconstrucao geodesica
f = np.asarray([0,0,1,3,3,7,7,7,7,5,2,1,1])
g = np.asarray([0,0,1,2,2,2,5,2,2,2,2,1,1])
geodesic_reconstruction_1d(f, g)

# watershed
img = mahotas.imread('dna.jpeg', as_grey=True)
img = img.astype(np.uint8)
#watershed(img)

# skeleton by influence zone (skiz)
img = mahotas.imread('bla.jpeg', as_grey=True)
img = img.astype(np.bool)
skiz = pymorph.skiz(img)
skiz *= 255 / skiz.max()
mahotas.imsave('skiz.png', skiz)
pylab.imshow(skiz)
pylab.show()
