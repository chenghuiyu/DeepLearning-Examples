import numpy as np
import matplotlib.pyplot as plt
from basic_linear_model import LinearModel


if __name__ == '__main__':
    x_train = [0.04, 0.08, 0.12]
    y_train = [0.2, 0.4, 0.6]

    W = 1.0
    weight_list = []
    mse_list = []
    linear_model = LinearModel(x_train, y_train)
    for W in np.arange(0.01, 1.1, 0.01):
        print("Weight = \n", W)
        loss_sum = linear_model.predict(W)
        mse = loss_sum / 3
        print("MSE: \n", mse)
        weight_list.append(W)
        mse_list.append(mse)

    plt.plot(weight_list, mse_list)
    plt.xlabel("weight")
    plt.ylabel("mse")
    plt.show()