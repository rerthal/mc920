import numpy as np
import cv2
import sys

def direction(window):
    angles = []
    direction000 = np.array(window) * np.array([[0,0,0], [1,1,1], [0,0,0]])
    direction045 = np.array(window) * np.array([[0,0,1], [0,1,0], [1,0,0]])
    direction090 = np.array(window) * np.array([[0,1,0], [0,1,0], [0,1,0]])
    direction135 = np.array(window) * np.array([[1,0,0], [0,1,0], [1,0,0]])
    angles.append((0, np.var(direction090) - np.var(direction000), direction000))
    angles.append((45, np.var(direction135) - np.var(direction045), direction045))
    angles.append((90, np.var(direction000) - np.var(direction090), direction090))
    angles.append((135, np.var(direction045) - np.var(direction135), direction135))
    angles.sort(key = lambda e: e[1], reverse=True)
    return angles[0]

if __name__ == '__main__':
    src = cv2.imread(sys.argv[1])
    src = cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
    filtered = np.ones(src.shape)
    for i in range(src.shape[0]/3):
        for j in range(src.shape[1]/3):
            slice = src[i*3:(i+1)*3, j*3:(j+1)*3]
            mask = direction(slice)[2]
            for ki in range(3):
                for kj in range(3):
                    filtered[i*3 + ki][j*3 + kj] = mask[ki][kj] * 255
    cv2.imwrite(sys.argv[1].split('.')[0] + '.jpg', src)
    cv2.imwrite(sys.argv[1].split('.')[0] + '_filtered_window_3.jpg', filtered)
