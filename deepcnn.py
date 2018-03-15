import tensorflow as tf
import numpy as np
import pickle

batch_size = 10


def get_training_xy():
    pickle_file = open("training_obj.pickle", "rb")
    data = pickle.load(pickle_file)
    x = []
    y = []
    for example in data:
        x.append(example["features"])

    for answers in data:
        y.append([answers["label"]])

    return x, y


def get_test_xy():
    pickle_file = open("test_obj.pickle", "rb")
    data = pickle.load(pickle_file)
    x = []
    y = []
    for example in data:
        x.append(example["features"])

    for answers in data:
        y.append(answers["label"])

    return x, y


def init_weights(shape):
    init_random_list = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_list)


def init_bias(shape):
    init_bias_vals = tf.constant(value=0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    #print(shape[0])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.add(tf.matmul(input_layer, W), b)


x_train_data, y_train_data = get_training_xy()
x_test_data, y_test_data = get_test_xy()

#x_train_data = np.array(x_train_data)
#x_train_data = np.reshape(x_train_data, [800, 64, 64, 3])

x = tf.placeholder(tf.float32, shape=[None, 64*64*3])
y_true = tf.placeholder(tf.float32, shape=[None, None, 2])
y_test_true = tf.placeholder(tf.float32, shape=[None, 2])
x_image = tf.reshape(x, [-1, 64, 64, 3])

convo_1 = convolutional_layer(x_image, shape=[16, 16, 3, 32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 16*16*64])
full_layer_1 = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout, 2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 20

record = 0
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    start = 0
    end = batch_size
    for i in range(steps):
        if i % 10 == 0:
            end = batch_size

        batch_x, batch_y = x_train_data[start:end], y_train_data[start:end]
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.99})
        end += batch_size

        if i % 1 == 0:
            print("Currently on step ", i+1)
            print("Accuracy : ")
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            accuracy = sess.run(acc, feed_dict={x: x_test_data, y_test_true: y_test_data, hold_prob: 0.99})
            print(accuracy)
            if accuracy > record:
                print("Saving..")
                saver.save(sess, "models/not_hot_dog.ckpt")
                record = accuracy
            print("\n")







