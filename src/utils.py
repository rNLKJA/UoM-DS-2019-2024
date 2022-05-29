import numpy as np

class Evaluation:
    """
    Custom evaluation metrics and functions
    - accuracy
    - r2 score
    - mean square error
    """
    
    @staticmethod
    def accuracy(true, pred):
        return np.sum(true == pred) / len(true)
    
    @staticmethod
    def r2_score(true, pred):
        return np.power(np.corrcoef(true, pred)[0, 1], 2)
    
    @staticmethod
    def mean_squared_error(true, pred):
        return np.mean(np.power((true - pred), 2))
    
    @staticmethod
    def mse(true, pred):
        self.mean_square_error(true, pred)
    
class Activation:
    """
    Custom activation functions
    """
    ...
    
class Distance:
    @staticmethod
    def Euclidean(x, y):
        return np.sqrt(np.power(np.sum(x - y), 2))