##############################################################################

import numpy as np

from sklearn.model_selection import cross_val_score


class Experimentation:
    """This class holds a ML model and allows training, predicting and
    evaluation with it.
    """

    def __init__(self, model, n_folds):
        """Constructor, holds the ML model."""

        self.model = model
        self.n_folds = n_folds

    def _cross_validation_score(self, X, y):
        """Makes cross_validation and evaluate each fold score."""

        return cross_val_score(self.model, X, y = y, cv = self.n_folds, 
                               score = 'f1_micro' ,verbose = 3, n_jobs = -1)

    def experiment_model(self, X, y):
        """Returns the results of the cross-validation."""

        folds_scores = self._cross_validation_score(X, y)
        mean_score = np.mean(folds_scores)
        std_score = np.std(folds_scores)

        return folds_scores, mean_score, std_score

    def predict_labels(self, X):
        """Returns the predicted labels for each row in X."""

        return self.model.predict(X)

    def predict_probabilities(self, X):
        """Returns the predicted probabilites for each row in X."""

        return self.model.predict_proba(X)

    def train_model(self, X, y):
        """Trains model with the list of features X, and the list 
        of labels y.
        """

        self.model.fit(X, labels)