import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import LabelBinarizer


def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    return X_train, y_train, X_test, y_test


def generate_indexes(X, n_clients):
    indexes = np.arange(np.random.randint(int(X.shape[0] * 0.8), X.shape[0]))

    np.random.shuffle(indexes)
    indexes = np.array_split(indexes, n_clients)

    clients_index = np.arange(n_clients)
    np.random.shuffle(clients_index)

    half_clients = int(n_clients / 2)
    samples = np.random.random_sample((half_clients,))
    for i in range(half_clients):
        client_A = indexes[clients_index[i]]
        client_B = indexes[clients_index[i + half_clients]]

        client_A = np.concatenate((client_A, client_B[:int(client_B.shape[0] *
                                                           samples[i])]))
        client_B = client_B[int(client_B.shape[0] * samples[i]):]

        indexes[clients_index[i]] = client_A
        indexes[clients_index[i + half_clients]] = client_B

    return indexes


def create_clients():
    n_clients = 6
    X, y, _, _ = load_data()

    indexes = generate_indexes(X, n_clients)

    X_slices = np.array([X[indexes[i]] for i in range(n_clients)],
                        dtype='object')
    y_slices = np.array([y[indexes[i]] for i in range(n_clients)],
                        dtype='object')
    return X_slices, y_slices
