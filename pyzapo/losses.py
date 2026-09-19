import numpy as np

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class BCELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true

        loss = (y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))

        return -np.mean(loss)

    def backward(self):
        y_pred = self.y_pred
        y_true = self.y_true

        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
