import numpy as np
import sys, os
from dataset.mnist import load_mnist
from PIL import Image
import matplotlib.pyplot as plt
import pickle

from ActivationFunction import *


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network  = pickle.load(f)

    print(type(network) , " = pickle type")
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = Sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = Sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = SoftMax(a3)

    return y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0

    batch_size = 100

    # for i in range(0, len(x), batch_size):
    #     y_batch = predict(network, x[i : i + batch_size])
    #     p = np.argmax(y_batch, axis=1)
    #     accuracy_cnt += np.sum(p == t[i : i + batch_size])
    #
    # print("Accuracy = " + str(float(accuracy_cnt) / len(x)))

    print(np.log(0.01))




