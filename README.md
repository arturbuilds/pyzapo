# PyZapo

PyZapo is a lightweight, modular, object-oriented deep learning framework built completely from scratch using only **NumPy**. It mimics the intuitive style of PyTorch, making it perfect for understanding modern neural network mechanics under the hood.

## 🚀 Features

- **Custom Autograd & Tensor Engine (New in 1.1.0):** Supports computational graphs with automated backward passes for seamless gradient tracking via `Tensor` and evaluation controls via `no_grad`.
- **Object-Oriented Architecture:** Build clean pipelines using `Sequential` blocks and custom `Module` classes.
- **Robust Built-in Optimization:** Every linear layer comes equipped with an integrated **Adam optimizer** right out of the box.
- **Gradient Saturation Protection:** Custom math safety controls (including vector clipping) in activations like `Sigmoid` to prevent gradient death.
- **Advanced Loss Functions:** Supports `MSELoss`, `BCELoss` (with automatic stabilization), and full multi-class **`CrossEntropyLoss`** (Fused Softmax + CE).

## 📦 Installation

Install the package directly from PyPI:

```bash
pip install pyzapo
```

## 🛠️ Quick Start: Multi-Class Digit Classification (0-9)

Here is how easily you can build, train, and test a deep neural network using **PyZapo 1.1.0** to classify inputs into 10 different categories (perfectly suited for datasets like MNIST):

```python
import numpy as np
import pyzapo as pz

# 1. Generate synthetic dataset (e.g., 5 flattened 28x28 images)
X = np.random.rand(5, 784)
# One-hot encoded labels for 10 distinct classes (digits 0-9)
y = np.eye(10)[:5]

# 2. Define deep network architecture (Clean OOP Style without .forward())
model = pz.Sequential([
    pz.Linear(784, 32),
    pz.ReLU(),
    pz.Linear(32, 10)
])

criterion = pz.CrossEntropyLoss()

# 3. Training Loop with automated backpropagation and Adam steps
print('Training the model...')
for epoch in range(1501):
    logits = model(X)           # Clean call syntax
    loss = criterion(logits, y) # Fused Softmax + CrossEntropy
    
    loss_grad = criterion.backward()
    model.backward(loss_grad)
    model.step(lr=0.01)

    if epoch % 300 == 0:
        print(f'Epoch {epoch:4d} | Loss: {loss:.6f}')

# 4. Inference and Evaluation using no_grad context
with pz.no_grad():
    test_logits = model(X)
    
shift_logits = test_logits - np.max(test_logits, axis=-1, keepdims=True)
probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits), axis=-1, keepdims=True)

predictions = np.argmax(probs, axis=-1)
true_classes = np.argmax(y, axis=-1)

print(f'True Labels: {true_classes.tolist()}')
print(f'Predictions: {predictions.tolist()}')
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
