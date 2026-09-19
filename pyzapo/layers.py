import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros((1, out_features))

        self.m_W, self.m_b = np.zeros_like(self.W), np.zeros_like(self.b)
        self.v_W, self.v_b = np.zeros_like(self.W), np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self.x = x

        return np.dot(x, self.W) + self.b

    def backward(self, delta):
        self.dW = np.dot(self.x.T, delta)
        self.db = np.sum(delta, axis=0, keepdims=True)

        next_delta = np.dot(delta, self.W.T)

        return next_delta

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db

        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW ** 2)
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db ** 2)

        m_W_corrected = self.m_W / (1 - beta1 ** self.t)
        m_b_corrected = self.m_b / (1 - beta1 ** self.t)
        v_W_corrected = self.v_W / (1 - beta2 ** self.t)
        v_b_corrected = self.v_b / (1 - beta2 ** self.t)

        self.W -= lr * m_W_corrected / (np.sqrt(v_W_corrected) + eps)
        self.b -= lr * m_b_corrected / (np.sqrt(v_b_corrected) + eps)

    def __call__(self, x):
        return self.forward(x)

class Sigmoid:
    def forward(self, x):
        self.y_pred = 1 / (1 + np.exp(-x))
        
        return self.y_pred

    def backward(self, delta):
        return delta * (self.y_pred * (1 - self.y_pred))

    def __call__(self, x):
        return self.forward(x)

class Sequential:
    def __init__(self, layers_list):
        self.layers = layers_list

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, loss_gradient):
        delta = loss_gradient
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        return delta

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr)

    def save_weights(self, filepath: str):
        weight_dict = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weight_dict[f'layer_{idx}_W'] = layer.W
                weight_dict[f'layer_{idx}_b'] = layer.b

        np.savez(filepath, **weight_dict)

    def load_weights(self, filepath: str):
        try:
            data = np.load(filepath)
            for idx, layer in enumerate(self.layers):
                if hasattr(layer, 'W'):
                    w_key = f'layer_{idx}_W'
                    b_key = f'layer_{idx}_b'

                    if w_key in data and b_key in data:
                        layer.W = data[w_key]
                        layer.b = data[b_key]

        except FileNotFoundError:
            print(f'Файл {filepath} не найден! Проверь путь.')

    def __call__(self, x):
        return self.forward(x)

class ReLU:
    def forward(self, x):
        self.x = x

        return np.maximum(0, x)

    def backward(self, delta):
        return delta * (self.x > 0)

    def __call__(self, x):
        return self.forward(x)

class Module:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def step(self, lr):
        for attr in vars(self).values():
            if hasattr(attr, 'step'):
                attr.step(lr)

    def save_weights(self, filepath: str):
        weights_dict = {}
        for name, attr in vars(self).items():
            if hasattr(attr, 'W'):
                weights_dict[f'{name}_W'] = attr.W
                weights_dict[f'{name}_b'] = attr.b
        np.savez(filepath, **weights_dict)

    def load_weights(self, filepath: str):
        try:
            data = np.load(filepath)
            for name, attr in vars(self).items():
                if hasattr(attr, 'W'):
                    if f'{name}_W' in data and f'{name}_b' in data:
                        attr.W = data[f'{name}_W']
                        attr.b = data[f'{name}_b']
        except FileNotFoundError:
            print(f'Файл {filepath} не найден!')
