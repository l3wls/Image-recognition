import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self._classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for i, c in enumerate(self._classes):
            # One-vs-All: current class is +1, all others are -1
            y_binary = np.where(y == c, 1, 0)

            for _ in range(self.n_iters):
                for index, x_i in enumerate(X):
                    y_predicted = np.dot(x_i, self.weights[i]) + self.bias[i]
                    y_predicted = 1 if y_predicted >= 0 else 0

                    update = self.lr * (y_binary[index] - y_predicted)
                    self.weights[i] += update * x_i
                    self.bias[i] += update

    def predict(self, X):
        # Score each class and pick the highest
        scores = np.dot(X, self.weights.T) + self.bias
        return self._classes[np.argmax(scores, axis=1)]
