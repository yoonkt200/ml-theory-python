import numpy as np
from sklearn import datasets
from keras.utils import np_utils
from keras import datasets

'''

    util functions define

'''


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# softmax function
def softmax(x_data):
    predictions = x_data - (x_data.max(axis=1).reshape([-1, 1]))
    softmax = np.exp(predictions)
    softmax /= softmax.sum(axis=1).reshape([-1, 1])
    return softmax

# cost function
def cross_entropy(z, y_data, reg_strength, W):
    sample_size = y_data.shape[0]
    cost = -np.log(z[np.arange(len(z)), np.argmax(y_data, axis=1)]).sum()
    cost /= sample_size
    cost += (reg_strength * (W**2).sum()) / 2
    return cost

# relu 함수 정의
def relu(z):
    return np.maximum(0, z)


'''

    Artificial Neural Network Class    

'''


# ANN Class
class ANN:
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=1000, verbose=False, reg_strength=1e-5, hidden_layer_depth=1):
        self._learning_rate = learning_rate  # 학습 계수
        self._max_iterations = max_iterations  # 반복 횟수
        self._threshold = threshold  # 학습 중단 계수
        self._verbose = verbose  # 중간 진행사항 출력 여부
        self._reg_strength = reg_strength # 정규화 파라미터 계수
        self._hidden_layer_depth = hidden_layer_depth # hidden layer 개수
        self._params = {} # 레이어별 학습 계수 dictionary

    # hidden layer 추가 함수
    def add_hidden_layer(self, layer_number, input_size, output_size):
        self._params['W' + str(layer_number)] = np.random.randn(input_size, output_size)
        self._params['b' + str(layer_number)] = np.zeros(output_size)

    # gradient 계산
    def gradient(self, x_data, y_pred, y_data):
        sample_size = y_data.shape[0]
        dy = (y_pred - y_data) / sample_size
        a = np.dot(x_data, self._params['W0']) + self._params['b0']
        grads = {}

        for j in range(0, self._hidden_layer_depth + 1):
            W_key = 'W' + str(j)
            b_key = 'b' + str(j)
            W_next_key = 'W' + str(j + 1)

            # # 마지막 activation : softmax
            if j == self._hidden_layer_depth:
                # softmax gradient : regularized
                t = np.dot(x_data, self._params['W0']) + self._params['b0']
                t = np.dot(t, self._params['W1']) + self._params['b1']
                gradient = np.dot(t.T, dy)
                # gradient = np.dot(x_data.transpose(), dy)
                # gradient += self._reg_strength * self._params[W_key]

                # layer gradient
                grads[W_key] = gradient
                grads[b_key] = np.sum(dy, axis=0)

            # 레이어간 activation : relu
            else:
                da = np.dot(dy, self._params[W_next_key].T)
                a = np.dot(x_data, self._params[W_key]) + self._params[b_key]

                # relu local gradient
                a[a >= 0] = 1
                a[a < 0] = 0
                gradient = a
                dz = gradient * da

                # layer gradient
                grads[W_key] = np.dot(x_data.transpose(), dz)
                grads[b_key] = np.sum(dz, axis=0)

        return grads

    # learning
    def fit(self, x_data, y_data):
        input_size = x_data.shape[1]

        # 가중계수 초기화
        self._params['W0'] = np.random.randn(input_size, len(self._params['W1']))
        self._params['b0'] = np.zeros(len(self._params['W1']))

        # print(self._params['W0'])
        # print(self._params['b0'])
        # print(self._params['W1'])
        # print(self._params['b1'])

        # learning
        for i in range(self._max_iterations):
            z = np.dot(x_data, self._params['W0']) + self._params['b0']
            z = relu(z)

            # feed forward
            for j in range(1, self._hidden_layer_depth + 1):

                # 마지막 activation : softmax
                if j == self._hidden_layer_depth:
                    z = np.dot(z, self._params['W' + str(j)]) + self._params['b' + str(j)]
                    z = softmax(z)

                # 레이어간 activation : relu
                else:
                    z = np.dot(z, self._params['W' + str(j)]) + self._params['b' + str(j)]
                    z = relu(z)

            # 최종 cost
            cost = cross_entropy(z, y_data, self._reg_strength, self._params['W1'])

            # 각 레이어별 local gradient
            grad = self.gradient(x_data, z, y_data)

            # 레이어별 local gradient를 이용하여 레이어별 계수 업데이트
            for j in range(1, self._hidden_layer_depth + 1):
                W_key = 'W' + str(j)
                b_key = 'b' + str(j)
                self._params[W_key] -= self._learning_rate * grad[W_key]
                self._params[b_key] -= self._learning_rate * grad[b_key]

            # 판정 임계값에 다다르면 학습 중단
            if cost < self._threshold:
                return False

            # 100 iter 마다 cost 출력
            if (self._verbose == True and i % 100 == 0):
                print ("Iter(Epoch): %s, Loss: %s" % (i, cost))

    # predict
    def predict(self, x_data):
        a = np.dot(x_data, self._params['W0']) + self._params['b0']
        z = relu(a)

        for j in range(0, self._hidden_layer_depth + 1):
            W_key = 'W' + str(j)
            b_key = 'b' + str(j)
            W = self._params[W_key]
            b = self._params[b_key]

            # # 마지막 activation : softmax
            if j == self._hidden_layer_depth:
                y = softmax(a)
                y = np.argmax(y, 1)
                return y

            # 레이어간 activation : relu
            else:
                a = np.dot(z, W) + b

        return False


# run example
if __name__ == "__main__":
    # iris = datasets.load_iris()
    # X = iris.data[:, :]
    # num = np.unique(iris.target, axis=0)
    # num = num.shape[0]
    # y = np.eye(num)[iris.target].astype(int)
    #
    # num_examples, num_features = np.shape(X)
    # num_classes = y.shape[1]

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X = X_train[:100, :]

    # one-hot encoding
    y_train = np_utils.to_categorical(y_train)
    y = y_train[:100, :]

    ann = ANN(learning_rate=0.001, threshold=0.01, max_iterations=2000, verbose=True, reg_strength=1e-8, hidden_layer_depth=1)
    ann.add_hidden_layer(1, 4, 10)
    ann.fit(X, y)

    # training score
    pred = ann.predict(X)
    num = 3
    pred = np.eye(num)[pred].astype(int)

    acc = 0
    for idx in range(len(y)):
        if (y[idx] == pred[idx]).all():
            acc += 1

    print('acc :', str(acc / len(y)))


