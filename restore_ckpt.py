from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
import tensorflow as tf
def _bias_variable(name,shape,initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var
def _weight_variable(name,shape,std):
    return _bias_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=std,dtype=tf.float32),
                          )
def inference(input):
    with tf.variable_scope('layer3') as scope:
        weights = _weight_variable('weights',[128,64],std=0.001)
        bias = _bias_variable('bias',[64],tf.constant_initializer(0.0))
        layer3 = tf.nn.relu(tf.matmul(input, weights) + bias, name=scope.name)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _weight_variable('weights', [64, 10], std=1 / 192.0)
        bias = _bias_variable('bias', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(layer3, weights), bias, name=scope.name)
    return softmax_linear

def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1),logits=logits,name = 'cross_entropy')
    cross_entropy_mean  = tf.reduce_mean(cross_entropy,name = 'cross_entropy')
    return cross_entropy_mean


batch_size = 100
training_epoch = 20
with tf.Graph().as_default() as g:
    saver = tf.train.import_meta_graph('./my_test_model-15.meta')
    x_place = g.get_tensor_by_name('input/x:0')
    y_place = g.get_tensor_by_name('input/y:0')
    weight_test = g.get_tensor_by_name('layer1/weights:0')
    layer2 = g.get_tensor_by_name('layer2/layer2:0')
    layer2 = tf.stop_gradient(layer2,name='stop_gradient')
    soft_result = inference(layer2)
    cost = loss(soft_result,y_place)
    opt = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_place, 1), tf.argmax(soft_result, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
with tf.Session(graph=g) as sess:
    value=[]
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        for _ in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(opt, feed_dict={x_place: batch_xs, y_place: batch_ys})
            cost_ = sess.run(cost, feed_dict={x_place: batch_xs, y_place: batch_ys})
            weight_test_value = sess.run(weight_test,feed_dict={x_place: batch_xs, y_place: batch_ys})
        print(("%s epoch: %d,cost: %.6f") % (datetime.now(), epoch + 1, cost_))
        if (epoch+1) % 5 == 0:
            value.append(weight_test_value)
    for i in range(len(value)-1):
        if value[i].all()==value[i+1].all():
            print("weight is equal")
    mean_accuary = sess.run(accuracy, {x_place: mnist.test.images, y_place: mnist.test.labels})
    print("accuracy %3.f" % mean_accuary)

