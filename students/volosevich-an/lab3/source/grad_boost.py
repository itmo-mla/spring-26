# a_t(x) = a_(t-1)(x) + η * α_t * b_t(x)
# b_t — приближение антиградиента
# α_t — одномерная минимизация
# Loss: multinomial logistic loss

import numpy as np
from decision_tree import DecisionTreeRegressor


class GradientBoostingClassifierCustom:
    def __init__(
        self,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_split=10,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.models = []
        self.alphas = []

        self.n_classes = None

        np.random.seed(self.random_state)

    def _softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def _log_loss(self, y_one_hot, probs):
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))


    # α_t = argmin L(F + α * update)
    def _find_best_alpha(self, F, update, y_one_hot):
        best_alpha = 0.0
        best_loss = float("inf")

        # practical line search
        for alpha in np.linspace(0.01, 1.0, 100):
            new_F = F + alpha * update
            probs = self._softmax(new_F)

            loss = self._log_loss(y_one_hot, probs)

            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha

        return best_alpha

    def fit(self, X, y):
        n_samples = X.shape[0]

        # classes
        self.n_classes = len(np.unique(y))

        # one-hot labels
        y_one_hot = np.eye(self.n_classes)[y]

        # Initial approximation a0 = 0
        F = np.zeros((n_samples, self.n_classes))

        self.models = []
        self.alphas = []

        for estimator_idx in range(self.n_estimators):

            sample_size = max(1, int(self.subsample * n_samples))

            sample_indices = np.random.choice(
                n_samples,
                sample_size,
                replace=False
            )

            X_sub = X[sample_indices]
            y_sub = y_one_hot[sample_indices]

            probs = self._softmax(F)
            probs_sub = probs[sample_indices]

            # Store trees for current boosting stage
            trees_stage = []

            stage_update = np.zeros((n_samples, self.n_classes))


            for k in range(self.n_classes):

                # Anti-gradient:
                # g_i = y_ik - p_ik
                residuals = y_sub[:, k] - probs_sub[:, k]

                # Newton-Raphson correction:
                # gamma = residual / Hessian
                hessian = probs_sub[:, k] * (1 - probs_sub[:, k]) + 1e-10

                target = residuals / hessian

                # Train tree on corrected pseudo-residuals
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )

                tree.fit(X_sub, target)

                # Predict on full dataset
                predictions = tree.predict(X)

                stage_update[:, k] = predictions

                trees_stage.append(tree)

            alpha_t = self._find_best_alpha(F, stage_update, y_one_hot)

            # Update ensemble
            F += self.learning_rate * alpha_t * stage_update

            # Save stage
            self.models.append(trees_stage)
            self.alphas.append(alpha_t)

    # Probability prediction
    def predict_proba(self, X):
        n_samples = X.shape[0]

        F = np.zeros((n_samples, self.n_classes))

        for trees_stage, alpha_t in zip(self.models, self.alphas):

            stage_update = np.zeros((n_samples, self.n_classes))

            for k, tree in enumerate(trees_stage):
                stage_update[:, k] = tree.predict(X)

            F += self.learning_rate * alpha_t * stage_update

        return self._softmax(F)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    