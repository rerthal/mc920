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

def is_foreground(image):
    dscptr = descriptor(image)
    return classifier.predict([dscptr[0], dscptr[1], dscptr[2]]) == 1.0

if __name__ == '__main__':
    rlt = []
    skf = cross_validation.StratifiedKFold(y, k=10)
    for train_index, test_index in skf:
        classifier = SVC()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        predictions = [classifier.predict(i) for i in X_test]
        matches = 0
        for i in range(len(predictions)):
            if predictions[i][0] == y_test[i] : matches = matches + 1
        rlt.append(float(matches) / len(predictions))
    print np.mean(rlt), np.var(rlt)