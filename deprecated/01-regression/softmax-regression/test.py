import math
import numpy as np

def softmax_func(x_data):
    predictions = x_data - (x_data.max(axis=1).reshape([-1, 1]))
    softmax = np.exp(predictions)
    softmax /= softmax.sum(axis=1).reshape([-1, 1])
    return softmax

def cost_func(softmax, y_data):
    sample_size = y_data.shape[0]
    print(range(sample_size))
    log_likelihood = np.log(softmax[range(sample_size), np.argmax(y_data, axis=1)])
    cost = -(np.sum(log_likelihood) / sample_size)
   
    return cost

a = np.array([[0.3, 2.9, 4.0], [3.3, 0.9, 0.4], [0.1, 2.9, 0.5]])
my_y = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

y3 = softmax_func(a)

# print(a)
# print(y3)
# print(my_y)
# print(y3[np.arange(my_y.shape[0]), my_y])
# print(cost_func(y3, my_y))

# print("------")
# print(y3.shape)
# print(my_y.shape)
# print("------")





import math
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[0:3, :]
num = np.unique(iris.target, axis=0)
num = num.shape[0]
y = np.eye(num)[iris.target].astype(int)[0:3, :]

y4 = softmax_func(X)
aa = cost_func(y4, y)

print(X)
print(y)
print(aa)

print(y4)
y4[np.arange(len(y4)), np.argmax(y, axis=1)] -= 1
print(np.argmax(y, axis=1))
print(y4)