import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.get_logger().setLevel('INFO')


class MLP:
    def __init__(self):
        self.model = None
        self.lr = 0.01
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.optimizer = tf.keras.optimizers.SGD(lr=self.lr, decay=self.lr/100,
                                                 momentum=0.9)

    def build(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=256, activation='relu',
                                             input_shape=(784, )))
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer,
                           metrics=self.metrics)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=2, verbose=2)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def get_layers(self):
        return [layer for layer in self.model.layers
                if isinstance(layer, keras.layers.Dense)]

    def test_model(self, X, y):
        test_loss, test_accuracy = self.model.evaluate(X, y, verbose=0)
        return test_loss, test_accuracy
