import numpy as np
from sklearn import datasets


'''

- GD를 이용하여 Softmax regression 학습
- 참고 : https://deepnotes.io/softmax-crossentropy

'''

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=100000, verbose=False, reg_strength=1e-5):
        self._learning_rate = learning_rate  # 학습 계수
        self._max_iterations = max_iterations  # 반복 횟수
        self._threshold = threshold  # 학습 중단 계수
        self._verbose = verbose  # 중간 진행사항 출력 여부
        self._reg_strength = reg_strength # 정규화 파라미터 계수

    # theta(W) 계수들 return
    def get_coeff(self):
        return self._W

    # softmax function
    def softmax_func(self, x_data):
        predictions = x_data - (x_data.max(axis=1).reshape([-1, 1]))
        softmax = np.exp(predictions)
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        return softmax

        # prediction result example
        # [[0.01821127 0.24519181 0.73659691]
        # [0.87279747 0.0791784  0.04802413]
        # [0.05280815 0.86841135 0.0787805 ]]

    # cost function 정의
    def cost_func(self, softmax, y_data):
        sample_size = y_data.shape[0]

        # softmax[np.arange(len(softmax)), np.argmax(y_data, axis=1)]
        # --> 해당 one-hot 의 class index * 해당 유닛의 출력을 각 row(1개의 input row)에 대해 계산
        # --> (n, 1) 의 shape
        cost = -np.log(softmax[np.arange(len(softmax)), np.argmax(y_data, axis=1)]).sum() 
        cost /= sample_size
        cost += (self._reg_strength * (self._W**2).sum()) / 2
        return cost

    # gradient 계산 (regularized)
    def gradient_func(self, y_pred, x_data, y_data):
        sample_size = y_data.shape[0]

        # softmax cost function의 미분 결과는 pi−yi. -> dy에 input값을 내적

        # softmax가 계산된 matrix에서, (해당 one-hot 의 class index * 해당 유닛)에 해당하는 유닛 위치에 -1을 더해줌.
        # softmax[np.arange(len(softmax)), np.argmax(y_data, axis=1)] -= 1
        # gradient = np.dot(x_data.transpose(), softmax) / sample_size

        dy = (y_pred - y_data) / sample_size
        gradient = np.dot(x_data.transpose(), dy)
        gradient += self._reg_strength * self._W
        return gradient

    # learning
    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)
        num_classes = y_data.shape[1]

        # 가중계수 초기화
        self._W = np.random.randn(num_features, num_classes) / np.sqrt(num_features / 2)

        for i in range(self._max_iterations):
            
            # y^ 계산
            z = np.dot(x_data, self._W)
            softmax = self.softmax_func(z)

            # cost 함수
            cost = self.cost_func(softmax, y_data)

            # softmax 함수의 gradient (regularized)
            gradient = self.gradient_func(softmax, x_data, y_data)

            # gradient에 따라 theta 업데이트
            self._W -= self._learning_rate * gradient

            # 판정 임계값에 다다르면 학습 중단
            if cost < self._threshold:
                return False

            # 100 iter 마다 cost 출력
            if (self._verbose == True and i % 100 == 0):
                print ("Iter(Epoch): %s, Loss: %s" % (i, cost))

    # prediction
    def predict(self, x_data):
        return np.argmax(x_data.dot(self._W), 1)


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :]
    num = np.unique(iris.target, axis=0)
    num = num.shape[0]
    y = np.eye(num)[iris.target].astype(int)

    # 학습 implementation
    model = SoftmaxRegression(learning_rate=0.1, threshold=0.01, max_iterations=1000, verbose=True)
    model.fit(X, y)
    
    # training score
    pred = model.predict(X)
    num = np.unique(pred, axis=0)
    num = num.shape[0]

    pred = np.eye(num)[pred].astype(int)

    acc = 0
    for idx in range(len(y)):
        if (y[idx] == pred[idx]).all():
            acc += 1

    print('acc :', str(acc / len(y)))

