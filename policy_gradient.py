import pickle

import numpy as np


class RMSProp:
    def __init__(
        self,
        lr,
        decay_rate=0.99,
    ):
        self.lr = lr
        self.decay_rate = decay_rate

    def step(self, params, grads, rmsprop):
        for i in range(len(params)):
            rmsprop[i] = (
                self.decay_rate * rmsprop[i] + (1 - self.decay_rate) * grads[i] ** 2
            )
            params[i] += self.lr * grads[i] / (np.sqrt(rmsprop[i]) + 1e-5)
            grads[i] *= 0


class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []

    def forward(self, x):
        y = self.softmax(x)
        return y

    def softmax(self, x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x

    def backward(self, discounted_epr, epdlogp):
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr
        return epdlogp

    def reset(self):
        None


class FullyConnected:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.rmsprop = [np.zeros_like(w)]
        self.x = []

    def forward(self, x):
        w = self.params[0]
        if len(self.x) != 0:
            self.x = np.vstack([self.x, np.array(x)])
        else:
            self.x = np.array(x)
        out = np.dot(x, w)
        return out

    def backward(self, dout):
        w = self.params[0]
        dx = np.dot(dout, w.T)
        dw = np.dot(self.x.T, dout)
        self.grads[0] += dw
        self.x = []
        # self.grads.reshape(dw.shape)
        return dx

    def reset(self):
        self.x = []


class ReLu:
    def __init__(self):
        self.params = []
        self.grads = []
        self.rmsprop = []
        self.mask = []

    def forward(self, x):
        mask = x <= 0
        if len(self.mask) != 0:
            self.mask = np.vstack([self.mask, mask])
        else:
            self.mask = mask
        out = x
        out[mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        self.mask = []
        return dx

    def reset(self):
        self.mask = []


class network:
    def __init__(self, input_size, midde, output_size):
        I, M, O = input_size, midde, output_size
        w1 = np.random.randn(I, M) / np.sqrt(I)
        w2 = np.random.randn(M, O) / np.sqrt(M)

        # -------------------------調整するパラメータ---------------------------------------

        self.gamma = 0.8  # 保管した報酬を時系列で関連づける
        self.decay_rate = 0.8  # RMSpropの移動平均をとる際のパラメータ
        self.batch_size = 30  # バッチサイズ
        self.optimizer = RMSProp(
            lr=0.02, decay_rate=self.decay_rate
        )  # 学習率のパラメータ

        # --------------------------------------------------------------------------------------------

        self.epsilon = 0.0
        self.drs = []
        self.dlogps = []

        self.layers = [FullyConnected(w1), ReLu(), FullyConnected(w2)]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads, self.rmsprop = [], [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            self.rmsprop += layer.rmsprop

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        y = self.loss_layer.forward(x)
        return y

    def backward(self):
        epdlogp = np.vstack(self.dlogps)
        epr = np.vstack(self.drs)
        self.dlogps, self.drs = [], []
        discounted_epr = self.discount_rewards(epr)
        dout = self.loss_layer.backward(discounted_epr, epdlogp)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def reset(self):
        self.dlogps, self.drs = [], []
        for layer in self.layers:
            layer.reset()

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def update(self):
        self.optimizer.step(self.params, self.grads, self.rmsprop)

    def record_reward(self, reward):
        self.drs.append(reward)

    def select_action(self, aprob):
        if np.random.random() < self.epsilon:
            action = np.random.choice(range(len(aprob)))
        else:
            action = np.random.choice(range(len(aprob)), p=aprob)
        y = [0] * 4
        y[action] = 1
        self.dlogps.append(y - aprob)
        return action

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.params, f)
