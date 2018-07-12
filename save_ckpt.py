from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',100,'''batch_size''')
tf.app.flags.DEFINE_integer('traing_epoches',15,'''epoch''')
tf.app.flags.DEFINE_string('check_point_dir','./','check_ponint_dir')
def _bias_variable(name,shape,initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var
def _weight_variable(name,shape,std):
    return _bias_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=std,dtype=tf.float32),
                          )
def inference(x):
    with tf.variable_scope('layer1') as scope:
        weights = _weight_variable('weights',[784,256],0.04)
        bias = _bias_variable('bias',[256],tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(x,weights)+bias,name=scope.name)
    with tf.variable_scope('layer2') as scope:
        weights = _weight_variable('weights',[256,128],std=0.02)
        bias = _bias_variable('bias',[128],tf.constant_initializer(0.2))
        layer2 = tf.nn.relu(tf.matmul(layer1,weights)+bias,name=scope.name)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _weight_variable('weights',[128,10],std=1/192.0)
        bias = _bias_variable('bias',[10],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(layer2,weights),bias,name=scope.name)
    return softmax_linear

def loss(logits,labels):
    print(labels.get_shape().as_list())
    print(logits.get_shape().as_list())
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1),logits=logits,name = 'cross_entropy')
    cross_entropy_mean  = tf.reduce_mean(cross_entropy,name = 'cross_entropy')
    return cross_entropy_mean

def train():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    softmax_linear = inference(x)
    cost = loss(softmax_linear,y)
    opt = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax_linear, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.traing_epoches):
            avg_cost = 0.0
            total_batch = int(mnist.train.num_examples/FLAGS.batch_size)
            for _ in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                sess.run(opt,feed_dict={x:batch_xs,y:batch_ys})
                cost_ = sess.run(cost,feed_dict={x:batch_xs,y:batch_ys})
            print(("%s epoch: %d,cost: %.6f")%(datetime.now(),epoch+1,cost_))
            if (epoch+1) % 5 == 0:
                check_point_file = os.path.join(FLAGS.check_point_dir,'my_test_model')
                saver.save(sess,check_point_file,global_step=epoch+1)
        mean_accuary = sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels})
        print("accuracy %3.f"%mean_accuary)
    print()

def main(_):
   train()


if __name__ == '__main__':
  tf.app.run()
