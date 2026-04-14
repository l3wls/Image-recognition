import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
       self.lr = learning_rate
       self.n_iters = n_iters
       self.activation_func = self._unit_step_func
       self.weights = None
       self.bias = None

    
