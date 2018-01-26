import numpy as np
import matplotlib.pyplot as plt
from manual_gradient import ManualGradient

if __name__ == '__main__':
    weight = np.random.rand(1)
    epochs = 10
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    manual_gradient = ManualGradient(x_data=x_data, y_data=y_data, weight=weight)
    for epoch in range(epochs):
        loss, w = manual_gradient.predict()
        print('Epoch: {0}, Loss is {1}'.format(epoch, loss))
