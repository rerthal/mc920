from descriptor import descriptor
import glob
import cv2

def to_descriptor(files):
    result = []
    for fingerprint in glob.glob(files):
        src = cv2.imread(fingerprint)
        src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        result.append(descriptor(src))
    return result

def foreground() : return to_descriptor("slices/foreground/*.jpg")

def background() : return to_descriptor("slices/background/*.jpg")