import pyzapo as pz
import numpy as np

class Ai(pz.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pz.Linear(2, 8)
        self.relu1 = pz.ReLU()
        self.fc2 = pz.Linear(8, 1)
        self.sigmoid = pz.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def backward(self, loss_gradient):
        delta = self.sigmoid.backward(loss_gradient)
        delta = self.fc2.backward(delta)
        delta = self.relu1.backward(delta)
        delta = self.fc1.backward(delta)
        return delta

if __name__ == '__main__':
    x = np.array([[5.0, 1.0], [1.0, 50.0], [4.5, 2.0], [2.0, 45.0]])
    y = np.array([[1.0], [0.0], [1.0], [0.0]])

    model = Ai()
    loss_fn = pz.BCELoss()

    print('Обучаем модель')

    for epoch in range(1, 2501):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss_grad = loss_fn.backward()
        model.backward(loss_grad)

        model.step(lr=0.01)

        if epoch % 500 == 0:
            print(f'Эпоха {epoch} -> Ошибка: {loss:.5f}')

    model.save_weights('model_oop.npz')
    print('-' * 50)

    fresh_model = Ai()
    fresh_model.load_weights('model_oop.npz')

    print('\nОтветы новой модели без обучения:')
    print(np.round(fresh_model(x), 2))
