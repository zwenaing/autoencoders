import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized
from scipy.sparse import csr_matrix

newsgroup = fetch_20newsgroups_vectorized(subset="train")
newsgroup_train_X = newsgroup['data']
file_path = 'tmp/newsgroup_autoencoder/'

# parameters
learning_rate = 0.1
num_steps = 30
batch_size = 256 
display_step = 10
examples_to_show = 10

# Network parameters
num_hidden = 100
num_input = 130107

X = tf.placeholder(tf.float32, [None, num_input])

weights = {
    "encoder_w1": tf.Variable(tf.random_normal([num_input, num_hidden])),
    "decoder_w1": tf.Variable(tf.random_normal([num_hidden, num_input])),
    "encoder_b1": tf.Variable(tf.random_normal([num_hidden])),
    "decoder_b1": tf.Variable(tf.random_normal([num_input])),
}

def encoder(X):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["encoder_w1"]), weights["encoder_b1"]))
    return layer_1


def decoder(encoded_X):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoded_X, weights["decoder_w1"]), weights["decoder_b1"]))
    return layer_1

with tf.name_scope("Encoding"):
    encode = encoder(X)

with tf.name_scope("Decoding"):
    decode = decoder(encode)

with tf.name_scope("Loss"):
    y_pred = decode
    y_true = X
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

with tf.name_scope("Optimizing"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, file_path)
    curr = 0
    newsgroup_train_X = csr_matrix.todense(newsgroup_train_X)
    for i in range(num_steps):
       curr += batch_size
       batch_x = newsgroup_train_X[:curr, :]
       _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x})
       print(i)
       if i % display_step == 0:
           print("Step: ", i, " Loss: " + "{:.3f}".format(loss_))
           saver.save(sess, file_path)

    train_encoded = sess.run(encode, feed_dict={X: newsgroup_train_X})
    np.save("encoded_train_X_100.npy",train_encoded)
