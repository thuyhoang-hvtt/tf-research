from __future__ import print_function, absolute_import, division, unicode_literals
from functools import *

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Create object holds data and load dataset
mnist = tf.keras.datasets.mnist
(feature_train, label_train), (feature_test, label_test) = mnist.load_data()
feature_train, feature_test = feature_train / 255.0, feature_test / 255.0
feature_train = feature_train[..., tf.newaxis]
feature_test = feature_test[..., tf.newaxis]

# Print more detail of data
print('Training image:\t{}'.format(feature_train.shape))
print('Testing image:\t{}'.format(feature_test.shape))

# Make shuffled datasets
train_ds = tf.data.Dataset.from_tensor_slices((feature_train, label_train)) \
    .shuffle(buffer_size=10000).batch(batch_size=32)

test_ds = tf.data.Dataset.from_tensor_slices((feature_test, label_test)) \
    .batch(batch_size=32)


# Class hold our Model
class ModelX(Model):
    def __init__(self):
        super(ModelX, self).__init__()
        self.architecture = list()
        self.architecture.append(Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.architecture.append(Flatten())
        self.architecture.append(Dense(units=128, activation='relu'))
        self.architecture.append(Dense(units=10, activation='softmax'))

    def __call__(self, x):
        return reduce(lambda output, layer: layer(output), self.architecture, x)


# ---- Configure our model ----
model = ModelX()


#  - Loss function
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

#  - Optimizer
optimizer = tf.keras.optimizers.Adam()

#  - Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Define function step in our pipeline
@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        tr_loss = loss_function(y_true=labels, y_pred=predictions)

    gradients = tape.gradient(target=tr_loss, sources=model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(tr_loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(features, labels):
    predictions = model(features)
    te_loss = loss_function(y_true=labels, y_pred=predictions)

    test_loss(te_loss)
    test_accuracy(labels, predictions)


# Main process

if __name__ == '__main__':
    print('Main processing..')
    EPOCHS = 5

    for epoch in range(EPOCHS):
        for features, labels in train_ds:
            train_step(features, labels)

        for test_features, test_labels in test_ds:
            test_step(test_features, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'

        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()