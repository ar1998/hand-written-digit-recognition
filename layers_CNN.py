import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


input_layer = tf.reshape(x,[-1,28,28,1])
    
conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5,5],
                            padding='same',
                            activation=tf.nn.relu)
    
pool_1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2)
    
conv2 = tf.layers.conv2d(inputs=pool_1,
                            filters=64,
                            kernel_size=[5,5],
                            padding='same',
                            activation=tf.nn.relu)
pool_2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2)
    
pool_to_flat = tf.reshape(pool_2,[-1,7*7*64])
dense_layer = tf.layers.dense(inputs=pool_to_flat,
                                 units=1024,
                                 activation=tf.nn.relu)
drop = tf.layers.dropout(inputs=dense_layer,rate=0.4)
    
output_layer = tf.layers.dense(inputs=drop,units=10)
    
    
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

steps = 3000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        x_batch,y_batch = mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:x_batch,y_true:y_batch})       
        if i%50 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            matches = tf.equal(tf.argmax(output_layer,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
            print('\n')
