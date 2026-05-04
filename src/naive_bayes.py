import numpy as np

class NaiveBayes:
    def fit(self, X, y, alpha=1):
        # X: feature matrix numeric data from feature extraction
        # y: labels 
        # alpha: smoothing in order to to avoid zero probabilities

        #  number of samples and features
        n_samples, n_features = X.shape 

        #  this method for trying to predict
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Store how common each class is ( prior probabilities)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Store probability of each feature being 1 for each class
        self._feature_probs = np.zeros((n_classes, n_features), dtype=np.float64)

        # Loop through each class
        for index, c in enumerate(self._classes):

            # Get only samples belonging to this class
            X_c = X[y == c]

            #  need class probability
           
            self._priors[index] = X_c.shape[0] / float(n_samples)

            #  FEATURE PROBABILITY
            # calculate how often each feature = 1 for this class
            # +alpha prevents probability from becoming 0 
            self._feature_probs[index, :] = (
                X_c.sum(axis=0) + alpha
            ) / (X_c.shape[0] + 2 * alpha)

    def predict(self, X):
        # Predict for all samples
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Predict for onee sample
        posteriors = []

        # Loop through each class
        for index, _ in enumerate(self._classes):

            
            # Use log to avoid very small numbers
            prior = np.log(self._priors[index])

           
            # Clip probabilities to avoid log(0) and make sure to have numerical stability
            p = np.clip(self._feature_probs[index], 1e-10, 1 - 1e-10)

            
            # If feature = 1  use log(p)
            # If feature = 0  use log(1 - p)
            likelihood = np.sum(
                x * np.log(p) + (1 - x) * np.log(1 - p)
            )

            
            # Combine prior + likelihood
            posteriors.append(prior + likelihood)

        
        # Pick class with highest probability
        return self._classes[np.argmax(posteriors)]