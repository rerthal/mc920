from sklearn.svm import SVC
from sklearn import cross_validation
from descriptor import descriptor
import numpy as np
import data

background_data = map(lambda e: [e[0], e[1], e[2]], data.background())
foreground_data = map(lambda e: [e[0], e[1], e[2]], data.foreground())
background_labels = map(lambda e: 0, background_data)
foreground_labels = map(lambda e: 1, foreground_data)

X = np.array(background_data + foreground_data)
y = np.array(background_labels + foreground_labels)

classifier = SVC(C = 1000)
classifier.fit(X, y)

def is_foreground(image, w=8):
    dscptr = descriptor(image, w)
    return classifier.predict([dscptr[0], dscptr[1], dscptr[2]]) == 1.0