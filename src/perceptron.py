import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        # Learning rate controls the size of weight updates
        self.lr = learning_rate

        # Number of training iterations over the dataset
        self.n_iters = n_iters

        # Model parameters (initialized during training)
        self.weights = None
        self.bias = None

        # Stores unique class labels
        self._classes = None

    def fit(self, X, y):
        # Determine number of features
        n_features = X.shape[1]

        # Identify all unique classes in the dataset
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize weights (one vector per class)
        self.weights = np.zeros((n_classes, n_features))

        # Initialize bias (one value per class)
        self.bias = np.zeros(n_classes)

        # Train one classifier per class (One-vs-All approach)
        for i, c in enumerate(self._classes):

            # Convert labels to binary:
            # current class = 1, all other classes = 0
            y_binary = np.where(y == c, 1, 0)

            # Repeat training for a fixed number of iterations
            for _ in range(self.n_iters):

                # Loop through each training example
                for index, x_i in enumerate(X):

                    # Compute linear score (weighted sum + bias)
                    y_predicted = np.dot(x_i, self.weights[i]) + self.bias[i]

                    # Apply threshold to convert score into binary prediction
                    y_predicted = 1 if y_predicted >= 0 else 0

                    # Compute update value based on prediction error
                    update = self.lr * (y_binary[index] - y_predicted)

                    # Update weights to improve future predictions
                    self.weights[i] += update * x_i

                    # Update bias to adjust decision boundary
                    self.bias[i] += update

    def predict(self, X):
        # Compute scores for all classes
        scores = np.dot(X, self.weights.T) + self.bias

        # Select class with highest score
        return self._classes[np.argmax(scores, axis=1)]