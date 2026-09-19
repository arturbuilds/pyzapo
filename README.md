# PyZapo

**PyZapo** is a lightweight, object-oriented Deep Learning framework written completely from scratch using **pure NumPy** (without PyTorch). 

It features an in-built adaptive **Adam optimizer** right inside the layers, making neural network training fast, clean, and highly efficient without any extra boilerplate code.

## Features
* **PyTorch-Style Syntax:** Built-in `__call__` allows you to invoke models and layers like functions (`model(X)`).
* **OOP Architecture:** Inherit from `Module` to build complex custom neural networks.
* **Built-in Adam Optimizer:** Adaptive learning rate for each weight out of the box.
* **Modern Activations:** High-performance `ReLU` and `Sigmoid` layers.
* **Loss Functions:** `MSELoss` and binary cross-entropy (`BCELoss`) with clipping safety.
* **Weights Management:** Save and load your trained models instantly with `.npz` binary files.

## Installation 📦
```bash
pip install pyzapo
```

## Quick Start 💻
```python
import numpy as np
import pyzapo as pz

# 1. Define your custom architecture
class MyCoolAi(pz.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pz.Linear(2, 8)
        self.relu = pz.ReLU()
        self.fc2 = pz.Linear(8, 1)
        self.sigmoid = pz.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def backward(self, loss_gradient):
        delta = self.sigmoid.backward(loss_gradient)
        delta = self.fc2.backward(delta)
        delta = self.relu.backward(delta)
        delta = self.fc1.backward(delta)
        return delta

# 2. Train and Save
X = np.array([[5.0, 1.0], [1.0, 50.0]])
y = np.array([[1.0], [0.0]])

model = MyCoolAi()
criterion = pz.BCELoss()

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    
    loss_grad = criterion.backward()
    model.backward(loss_grad)
    model.step(lr=0.01)

model.save_weights("my_model.npz")
```
