import numpy as np
from scipy import sparse


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False
    ):
        """
        Trains this classifier using stochastic gradient descent.

        Args:
            x: N x D array of training data. Each training point is a
              D-dimensional column.
            y: 1-dimensional array of length N with labels 0-1, for 2 classes.
            learning_rate: (float) learning rate for optimization.
            reg: (float) regularization strength.
            num_iters: (integer) number of steps to take when optimizing
            batch_size: (integer) number of training examples to use at each
              step.
            verbose: (boolean) If true, print progress during optimization.

        Returns:
            a list containing the value of the loss function at each training
            iteration.
        """
        # Add a column of ones to X for the bias sake.
        x = LogisticRegression.append_biases(x)
        num_train, dim = x.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim, 1) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        idxs = np.arange(num_train)
        for it in range(num_iters):
            ###################################################################
            # TODO:
            # Sample batch_size elements from the training data and their
            # corresponding labels to use in this round of gradient descent.
            # Store the data in X_batch and their corresponding labels in
            # y_batch; after sampling X_batch should have shape
            # (batch_size, dim) and y_batch should have shape (batch_size,)
            #
            # Hint: Use np.random.choice to generate indices. Sampling with
            # replacement is faster than sampling without replacement.
            ###################################################################
            batch_idxs = np.random.choice(idxs, size=batch_size)

            x_batch = x[batch_idxs]
            y_batch = y[batch_idxs]
            #################################################################
            #                       END OF YOUR CODE
            #################################################################
            # Evaluates loss and gradient
            loss, grad = self.loss(x_batch, y_batch, reg)
            self.loss_history.append(loss)
            # Performs parameter update
            #################################################################
            # TODO:
            # Update the weights using the gradient and the learning rate.
            #################################################################
            self.w -= learning_rate * grad
            #################################################################
            #                       END OF YOUR CODE
            #################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def predict_proba(self, x: np.ndarray, append_bias: bool = False):
        """
        Uses the trained weights of this linear classifier to predict
        probabilities for data points.

        Args:
            x: N x D array of data. Each row is a D-dimensional point.
            append_bias: bool. Whether to append bias before predicting or not.

        Returns:
            probabilities of classes for the data in X, 2-dimensional array with
            a shape (N, 2), and each row is a distribution of classes
            [prob_class_0, prob_class_1].
        """
        if append_bias:
            x = LogisticRegression.append_biases(x)
        #######################################################################
        # TODO:
        # Implement this method. Store the probabilities of classes in y_proba.
        # Hint: It might be helpful to use np.vstack and np.sum
        #######################################################################
        z = x @ self.w
        y_proba = sigmoid(z)
        #######################################################################
        #                           END OF YOUR CODE
        #######################################################################
        return y_proba

    def predict(self, x: np.ndarray):
        """
        Uses the ```predict_proba``` method to predict labels for data points.

        Args:
            x: N x D array of training data. Each column is a D-dimensional pt.

        Returns:
            predicted labels for the data in X. y_pred is a 1-dimensional array
            of length N, and each element is an integer giving the predicted
            class.
        """

        ######################################################################
        # TODO:
        # Implement this method. Store the predicted labels in y_pred.
        ######################################################################
        y_proba = self.predict_proba(x, append_bias=True)
        y_pred = y_proba > 0.5

        ######################################################################
        #                           END OF YOUR CODE
        ######################################################################
        return y_pred

    def loss(self, x_batch: np.ndarray, y_batch: np.ndarray, reg):
        """Logistic Regression loss function

        Args:
            x_batch: N x D array of data. Data are D-dimensional rows
            y_batch: 1-dimensional array of length N with labels 0-1, for 2
              classes
        Returns:
            a tuple of:
            - loss as single float
            - gradient with respect to weights w; an array of same shape as w
        """
        # Compute loss and gradient. Your code should not contain python loops.

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead, so we divide by num_train.
        # Note that the same thing must be done with gradient.

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.
        y_proba = self.predict_proba(x_batch, append_bias=False)
        y_batch_reshaped = y_batch[:, np.newaxis]
        loss = -(
            y_batch_reshaped * np.log(y_proba)
            + (1 - y_batch_reshaped) * np.log(1 - y_proba)
        ).mean()
        grad = (
            x_batch
            .multiply(y_proba - y_batch_reshaped)
            .mean(axis=0)
            .T
        )

        return loss, grad

    @staticmethod
    def append_biases(x: np.ndarray):
        return sparse.hstack((x, np.ones(x.shape[0])[:, np.newaxis])).tocsr()
