import numpy as np

class ManualGradient:

    def __init__(self, x_data, y_data, weight, alpha=0.01):
        if not isinstance(x_data, np.ndarray):
            try:
                x_data = np.array(x_data)
            except:
                raise Exception("training data must be array")
        if not isinstance(y_data, np.ndarray):
            try:
                y_data = np.array(y_data)
            except:
                raise Exception("training labels must be array")

        self.x_data = x_data
        self.y_data = y_data
        self.weight = weight
        self.alpha = alpha

    def _forward(self, W, x_trains):
        return np.multiply(W, x_trains)

    def _loss(self, W, x_train, y_train):

        result = self._forward(W, x_train)
        loss = np.power((result-y_train), 2)
        return loss

    def _gradient(self, W, x_train, y_train):
        gradient_result = 2 * np.multiply(x_train,
                                          (np.multiply(W, x_train) - y_train))

        return gradient_result

    def predict(self):
        w = 0.0
        for x, y in zip(self.x_data, self.y_data):
            w_gradient = self._gradient(self.weight, x ,y)
            w = w - self.alpha * w_gradient
            loss = self._loss(w, x, y)
        return loss, w







