import numpy as np
import torch

class TrainingData:

    def __init__(self, x, y):
        
        self._set_training_data(x, y)
   
   def _set_training_data(self, x, y):
       self.x, self.y = self._check_inputs(x, y)
       
   def _check_inputs(self, x, y):

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        if y.dim() == 1:
            y = y.reshape((-1,1))
        if x.shape[0] != y.shape[0]:
            x = x.reshape((y.shape[0], -1))

        return x, y
