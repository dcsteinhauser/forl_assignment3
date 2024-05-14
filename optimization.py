import numpy as np

#### This code has been adapted from https://github.com/qzed/irl-maxent

class Optimizer:

    def __init__(self):
        self.parameters = None

    def init(self, parameters):
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        raise NotImplementedError


class Initializer:
    """
    Base-class for an Initializer, specifying a strategy for parameter
    """
    def __init__(self):
        pass

    def initialize(self, shape):
        """
        Create an initial set of parameters.
        """
        raise NotImplementedError

    def __call__(self, shape):
        return self.initialize(shape)

class Constant(Initializer):
    """
    An Initializer, initializing parameters to a constant value.
    """
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        """
        Create set of parameters with initial fixed value.
        """
        if callable(self.value):
            return np.ones(shape) * self.value(shape)
        else:
            return np.ones(shape) * self.value

def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):

    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))

    return _lr

class ExponentiatedSGA(Optimizer):

    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters *= np.exp(lr * grad)

        if self.normalize:
            self.parameters /= self.parameters.sum()