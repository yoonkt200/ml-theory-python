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


# relu func
def relu(z):
    return np.maximum(0, z)


# relu local gradient
def relu_grad(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    gradient = x
    return gradient


'''

    Artificial Neural Network Class    

'''


# ANN Class
class ANN:
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=1000, verbose=False, reg_strength=1e-5, hidden_layer_depth=1):
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._threshold = threshold
        self._verbose = verbose
        self._reg_strength = reg_strength
        self._hidden_layer_depth = hidden_layer_depth
        self._params = {}

    # hidden layer add func
    def add_hidden_layer(self, layer_number, input_size, output_size):
        self._params['W' + str(layer_number)] = np.random.randn(input_size, output_size)
        self._params['b' + str(layer_number)] = np.zeros(output_size)

    # gradient calc
    def gradient(self, x_data, y_pred, y_data):
        sample_size = y_data.shape[0]
        dy = (y_pred - y_data) / sample_size
        a = np.dot(x_data, self._params['W0']) + self._params['b0']
        z = relu(a)
        grads = {}

        # iterate each layer
        for j in range(0, self._hidden_layer_depth + 1):
            W_key = 'W' + str(j)
            b_key = 'b' + str(j)
            W_next_key = 'W' + str(j + 1)

            # last activation : softmax gradient
            if j == self._hidden_layer_depth:
                t = np.dot(x_data, self._params['W0']) + self._params['b0']
                t = relu(t)
                gradient = np.dot(t.T, dy)

                # softmax gradient : regularized
                gradient += self._reg_strength * self._params[W_key]

                # layer gradient
                grads[W_key] = gradient
                grads[b_key] = np.sum(dy, axis=0)

            # input-hidden layer gradient
            elif j == 0:
                da = np.dot(dy, self._params[W_next_key].T)
                dz = relu_grad(a) * da
                grads[W_key] = np.dot(x_data.T, dz)
                grads[b_key] = np.sum(dz, axis=0)

            # between activation : relu gradient
            # if want n-hidden learning, fill in this chapter
            else:
                pass

        return grads

    # learning
    def fit(self, x_data, y_data):
        input_size = x_data.shape[1]

        # W init
        self._params['W0'] = np.random.randn(input_size, len(self._params['W1']))
        self._params['b0'] = np.zeros(len(self._params['W1']))

        # learning
        for i in range(self._max_iterations):
            z = np.dot(x_data, self._params['W0']) + self._params['b0']
            z = relu(z)

            # feed forward
            for j in range(1, self._hidden_layer_depth + 1):

                # last activation : softmax
                if j == self._hidden_layer_depth:
                    z = np.dot(z, self._params['W' + str(j)]) + self._params['b' + str(j)]
                    z = softmax(z)

                # between activation : relu
                else:
                    z = np.dot(z, self._params['W' + str(j)]) + self._params['b' + str(j)]
                    z = relu(z)

            # final cost
            cost = cross_entropy(z, y_data, self._reg_strength, self._params['W1'])

            # get each layer local gradient
            grad = self.gradient(x_data, z, y_data)

            # each layer W update by using local gradient
            for j in range(1, self._hidden_layer_depth + 1):
                W_key = 'W' + str(j)
                b_key = 'b' + str(j)
                self._params[W_key] -= self._learning_rate * grad[W_key]
                self._params[b_key] -= self._learning_rate * grad[b_key]

            # leaning cost threshold
            if cost < self._threshold:
                return False

            # print cost 100, 200... epoch
            if (self._verbose == True and i % 100 == 0):
                print ("Iter(Epoch): %s, Loss: %s" % (i, cost))

    # predict
    def predict(self, x_data):
        a = np.dot(x_data, self._params['W0']) + self._params['b0']
        z = relu(a)

        for j in range(1, self._hidden_layer_depth + 1):
            W_key = 'W' + str(j)
            b_key = 'b' + str(j)
            W = self._params[W_key]
            b = self._params[b_key]

            # last activation : softmax
            if j == self._hidden_layer_depth:
                a = np.dot(z, W) + b
                y = softmax(a)
                y = np.argmax(y, 1)
                return y

            # between activation : relu
            else:
                a = np.dot(z, W) + b
                z = relu(a)

        return False


# run example
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # mini batch
    train_size = X_train.shape[0]
    batch_size = 1000

    batch_mask = np.random.choice(train_size, batch_size)
    X_train = X_train[batch_mask]
    y_train = y_train[batch_mask]

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X = X_train[:, :] / 255.0

    # one-hot encoding
    y_train = np_utils.to_categorical(y_train)
    y = y_train[:, :]

    ann = ANN(learning_rate=0.05, threshold=0.01, max_iterations=10000, verbose=True, reg_strength=1e-8, hidden_layer_depth=1)
    ann.add_hidden_layer(1, 64, 10)
    ann.fit(X, y)


    '''

        training score

    '''


    pred = ann.predict(X)
    acc = 0
    for idx in range(len(y)):
        if np.argmax(y[idx]) == pred[idx]:
            acc += 1

    print('acc :', str(acc / len(y)))