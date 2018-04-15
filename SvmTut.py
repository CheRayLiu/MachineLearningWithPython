import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print(digits.data)
print(digits.target)
print(digits.images[0])

clf = svm.SVC(gamma = 0.0001, C =100)

x,y = digits.data[:-10], digits.target[:-10]

clf.fit(x,y)

data = digits.data[-4].reshape(1,-1)
print('Prediction:', clf.predict(data))

plt.imshow(digits.images[-4], cmap = plt.cm.gray_r, interpolation = "nearest")