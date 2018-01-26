import numpy as np


class LinearModel:

    def __init__(self, x_data, y_data):

        if not isinstance(x_data, np.ndarray):
            try:
                x_data = np.array(x_data)
            except:
                raise Exception("Training data should be array")

        if not isinstance(y_data, np.ndarray):
            try:
                y_data = np.array(y_data)
            except:
                raise Exception("Y data should be array")

        self.x_data = x_data
        self.y_data = y_data

    def _forward(self, W, x_train):

        return np.multiply(x_train, W)

    def _loss(self, W, x_train, y_train):
        """
        使用交叉熵进行损失函数的计算
        """
        result = self._forward(W, x_train)
        loss = - (np.multiply(y_train, np.log(result)) +
                  np.multiply((1 - y_train), np.log((1 - result))))

        return loss

    def predict(self, w):

        loss_sum = 0.0
        for x, y in zip(self.x_data, self.y_data):
            y_pred = self._forward(w, x)
            loss = self._loss(w, x, y)
            loss_sum += loss
            print("\t", x, y, y_pred, loss)

        return loss_sum