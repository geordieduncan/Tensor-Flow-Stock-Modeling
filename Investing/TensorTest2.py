import numpy as np
import tensorflow as tf
import pandas as pd
from Tkinter import *

def cost(a, b):
    return abs(a - b) ** 2


def BuildTester(s):
    data = pd.read_csv(s + '.csv')
    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]
    # Make data a numpy array
    data = data.values
    train_start = 0
    train_end = int(np.floor(n))
    test_start = train_end
    test_end = n
    data_train = data[np.arange(0, train_end - 1), :]
    data_test = data[np.arange(test_start -1 , test_end), :]
    # Scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    # Build X and y
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    # Model architecture parameters
    n_stocks = 10
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(cost(out, Y))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())

    # Number of epochs and batch size
    epochs = 150
    batch_size = 10
    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    # Print final MSE after Training
    y_estimate_scaled = net.run(out, feed_dict={X: X_test})
    mse_final = net.run(mse, feed_dict={X: X_train, Y: y_train})
    f = float(y_estimate_scaled - X_test[-1][0])
    return (f, mse_final)
