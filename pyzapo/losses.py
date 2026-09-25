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
        return ((self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))) / self.y_true.size
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class CrossEntropyLoss:
    def forward(self, logits, y_true):
        self.y_true = y_true

        vals_max = np.max(logits, axis=-1, keepdims=True)
    
        s = logits - vals_max
        e = np.exp(s)
        p = e / np.sum(e, axis=-1, keepdims=True)

        self.y_pred = p
        self.y_pred = np.clip(p, 1e-15, 1 - 1e-15)

        loss = -np.sum(y_true * np.log(self.y_pred)) / y_true.size

        return loss

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.size

    def __call__(self, logits, y_true):
        return self.forward(logits, y_true)
