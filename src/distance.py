import numpy as np


class Distance:
    @staticmethod
    def Minkowski(X, Y, p):
        """
        Calculate the Minkowski distance based on given p value.
        It follow the function: $D(X, Y) = {\sum^n_{i=1} |x_i - y_i|^p} ^ \frac{1}{p}$
        """
        # check X, Y data type, if X, Y have python type list, then convert the list to np.array
        if type(X) == list or type(Y) == list:
            X, Y = np.array(X), np.array(Y)

        return np.power(np.sum(np.abs(X - Y)), (1 / p))

    @classmethod
    def Manhatten(clf, X, Y):
        """
        Calculate the Manhatten distance via calling Minkowski distance with p value = 1
        It follow the function: $D(X, Y) = \sum^n_{i=1} |x_i - y_i|$
        """
        return clf.Minkowski(X, Y, p=1)

    @classmethod
    def Euclidean(clf, X, Y):
        """
        Calculate the Euclidean distance via calling Minkowski distance with p value = 2
        It follow the function: $D(X, Y) = {\sum^n_{i=1} |x_i - y_i|^2} ^ \frac{1}{2}$
        """
        return clf.Minkowski(X, Y, p=2)
