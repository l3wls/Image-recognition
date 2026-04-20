import numpy as np

class NaiveBayes:
    def fit(self, X, y, alpha=1):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._priors = np.zeros(n_classes, dtype=np.float64)
        self._feature_probs = np.zeros((n_classes, n_features), dtype=np.float64)

        for index, c in enumerate(self._classes):
            X_c = X[y == c]
            self._priors[index] = X_c.shape[0] / float(n_samples)
            # Bernoulli: P(feature=1 | class) with Laplace smoothing
            self._feature_probs[index, :] = (X_c.sum(axis=0) + alpha) / (X_c.shape[0] + 2 * alpha)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        for index, c in enumerate(self._classes):
            prior = np.log(self._priors[index])
            p = np.clip(self._feature_probs[index], 1e-10, 1 - 1e-10)
            # log-likelihood for Bernoulli features
            likelihood = np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))
            posteriors.append(prior + likelihood)

        return self._classes[np.argmax(posteriors)]
