import tensorflow as tf


class PolynomialRegression:
    """
    Polynomial regression will find the best [w2], [w1] and [b] for the function [Y = w2*X^2 + w1*X + b] by minimizing
    the total loss with the training data using gradient descent.
    """
    def __init__(self, x, y):
        """
        Initialize all variables given placeholders x and y.

        :param x: tf.placeholder
        :param y: tf.placeholder
        """
        self._prediction = None
        self._loss = None
        self._optimizer = None
        self.w2 = None
        self.w1 = None
        self.b = None
        self.x = x
        self.y = y
        pass

    def prediction(self):
        """
        Create the prediction with variables [w2], [w1] and [b] so that computes [Y = w2*X^2 + w1*X + b]

        :return: prediction
        """
        if self._prediction is None:
            self.w2 = tf.Variable(0.0, name='weight_2')
            self.w1 = tf.Variable(0.0, name='weight_1')
            self.b = tf.Variable(0.0, name='bias')
            self._prediction = tf.multiply(self.w2, tf.pow(self.x, 2))/100 + tf.multiply(self.w1, self.x) + self.b
        return self._prediction

    def loss(self):
        """
        Create an operation that calculates loss

        :return: loss
        """
        if self._loss is None:
            self._loss = tf.square(self.prediction() - self.y, name='loss')
        return self._loss

    def optimizer(self):
        """
        Create an optimizer with an operation that minimizes loss

        :return: optimizer
        """
        if self._optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss())
        return self._optimizer
