#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:56:47 2019

@author: jax
"""

from sklearn.datasets import fetch_mldata
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Get data

mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

# Divide into training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Shuffle the data
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# feature scaling
pipeline = StandardScaler().fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)


# Fit the model

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_transform)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
feature_columns=feature_columns)
dnn_clf.fit(x=X_train_transform, y=y_train.astype(int), batch_size=50, steps=40000)


# Evaluation
from sklearn.metrics import accuracy_score
y_pred = list(dnn_clf.predict(X_test_transform))
accuracy_score(y_test, y_pred)

dnn_clf.evaluate(X_test_transform,y_test.astype(int))


# =================== Try low level ===========================
import tensorflow as tf

# =============================== Construction phase =========================================
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


#def neuron_layer(X, n_neurons, name, activation=None):
#    with tf.name_scope(name):
#        n_inputs = int(X.get_shape()[1])
#        stddev = 2 / np.sqrt(n_inputs)
#        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
#        W = tf.Variable(init, name="weights")
#        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
#        z = tf.matmul(X, W) + b
#        if activation=="relu":
#            return tf.nn.relu(z)
#        else:
#            return z
#        
#
#with tf.name_scope("dnn"):
#    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
#    logits = neuron_layer(hidden2, n_outputs, "outputs")
    
    
# Tensorflow already have well_defined function
from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs",activation_fn=None)
  
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")
    
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
# initialize all variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()


 # ============================ Execution Phase =========================================    
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")   
    
n_epochs = 50
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels})
        
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    
    save_path = saver.save(sess, "./my_model_final.ckpt")

    
# =============== Use Neural Network to predict ==========================
with tf.Session() as sess:
    saver.restore(sess,"./my_model_final.ckpt")
    X_new_scaled = [...] # some new images (scaled from 0 to 1)
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    




