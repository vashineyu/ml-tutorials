"""model.py
Example Models
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.utils import to_categorical

class SimpleDNN(models.Model):
    def __init__(self, n_classes):
        super(SimpleDNN, self).__init__()
        self.layer1 = layers.Dense(units=128, activation="relu")
        self.layer2 = layers.Dense(units=64, activation="relu")
        self.output_layer = layers.Dense(units=n_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.output_layer(x)
        return out

class ConvNormActivation(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvNormActivation, self).__init__()
        self.conv = layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding="same",
            )
        self.norm = layers.BatchNormalization(axis=-1)
        self.acti = layers.Activation("relu")

    def call(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.norm(x)
        return self.acti(x)

class SimpleCNN(models.Model):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = ConvNormActivation(filters=64, kernel_size=3)
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = ConvNormActivation(filters=64, kernel_size=3)
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv3 = ConvNormActivation(filters=32, kernel_size=3)
        self.flat = layers.GlobalAveragePooling2D()
        self.output_layer = layers.Dense(units=n_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flat(self.conv3(x))
        out = self.output_layer(x)
        return out