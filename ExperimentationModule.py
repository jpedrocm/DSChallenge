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

    def _cross_validation_score(self, X, y, scoring):
        """Makes cross_validation and evaluate each fold score."""

        scores = cross_val_score(self.model, X, y = y, cv = self.n_folds,
                               scoring = scoring, verbose=2, n_jobs=1)

        return scores

    def experiment_model(self, X, y, scoring):
        """Returns the results of the cross-validation."""

        folds_scores = self._cross_validation_score(X, y, scoring)
        mean_score = np.mean(folds_scores)
        std_score = np.std(folds_scores)

        return folds_scores, mean_score, std_score

    def get_model(self):
        """Returns the ML model."""

        return self.model

    def predict_labels(self, X):
        """Returns the predicted labels for each row in X."""

        return self.model.predict(X)

    def predict_probs(self, X):
        """Returns the predicted probabilites for each row in X."""

        return self.model.predict_proba(X)

    def train_model(self, X, y):
        """Trains model with the list of features X, and the list 
        of labels y.
        """

        self.model.fit(X, y)