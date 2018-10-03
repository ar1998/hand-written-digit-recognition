import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def convolutional_layer(input_x,shape):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    b = tf.Variable(tf.constant(0.1,shape=[shape[3]]))
    layer = tf.nn.conv2d(input=input_x,filter=w,strides=[1,1,1,1],padding='SAME')
    return tf.nn.relu(layer + b)

def fully_connected_layer(input_x,n_dense):
    input_size = int(input_x.get_shape()[1])
    w = tf.Variable(tf.constant(0.1,shape=[input_size,n_dense]))
    b = tf.Variable(tf.constant(0.1,shape=[n_dense]))
    return tf.matmul(input_x,w)+b

def max_pool_2by2(x_input):
    return tf.nn.max_pool(x_input, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

conv_layer_1 = convolutional_layer(x_image,shape=[6,6,1,32])
pool_layer_1 = max_pool_2by2(conv_layer_1)

conv_layer_2 = convolutional_layer(pool_layer_1,shape=[6,6,32,64])
pool_layer_2 = max_pool_2by2(conv_layer_2)

flat_input = tf.reshape(pool_layer_2,[-1,7*7*64])
full_layer_1 = tf.nn.relu(fully_connected_layer(flat_input,1024))

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_1,keep_prob=hold_prob)

pred = fully_connected_layer(full_one_dropout,10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

steps = 3000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        x_batch,y_batch = mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:x_batch,y_true:y_batch,hold_prob:0.5})
        
        if i%50 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            matches = tf.equal(tf.argmax(pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')
 
