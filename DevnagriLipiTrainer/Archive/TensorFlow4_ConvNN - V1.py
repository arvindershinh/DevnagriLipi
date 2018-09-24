# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:26:40 2018

@author: Arvinder Shinh
"""
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()
import numpy as np


Num_Exp=100
a=np.random.randint(0,9,Num_Exp)

images=np.random.normal(size=Num_Exp*28*28*1).reshape(Num_Exp,28,28,1)

labels=np.zeros((Num_Exp,10),dtype=np.int32)
for i in range(Num_Exp-1):
    labels[i,a[i]]=1

#x=tf.placeholder(shape=(None,28,28,1),dtype=tf.float32, name='Images')
#y=tf.placeholder(shape=(None,10),dtype=tf.float32, name='Labels')
x=tf.constant(images, dtype=tf.float32, name='Images')
y=tf.constant(labels, dtype=tf.float32, name='Labels')

data=tf.data.Dataset.from_tensor_slices({'x': x, 'y': y})
data=data.shuffle(100).repeat().batch(5)

iterator=data.make_initializable_iterator()
batch=iterator.get_next()
ImageBatch=batch['x']
LabelBatch=batch['y']
#ImageBatch=x
#LabelBatch=y


'''Convolution Layer'''
def conv_layer(inputs, In, Out, name='conv'):
    with tf.name_scope(name):
        w=tf.Variable(tf.random_normal((5,5,In,Out)), dtype=tf.float32, name='w')
        b=tf.Variable(tf.zeros((Out)), dtype=tf.float32, name='b')
        conv=tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        activation=tf.nn.relu(conv+b)
        tf.summary.histogram(name+'_w_kernal', w)
        tf.summary.histogram(name+'_b_kernal', b)
        tf.summary.histogram(name+'_activation', activation)
        return activation
    
'''Pooling Layer'''
def pooling_layer(inputs, name='pooling'):
    with tf.name_scope(name):
        pooling=tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return pooling
    
'''Dense Layer'''
def dense_layer(inputs, In, Out, name='Dense'):
    with tf.name_scope(name):
        w=tf.Variable(tf.random_normal((In,Out)), dtype=tf.float32, name='w')
        b=tf.Variable(tf.zeros((Out)), dtype=tf.float32, name='b')
        dense=tf.matmul(inputs,w)
        activation=tf.nn.relu(dense+b)
        tf.summary.histogram(name+'_w_kernal', w)
        tf.summary.histogram(name+'_b_kernal', b)
        tf.summary.histogram(name+'_activation', activation)
        return activation



#In = [-1,28,28,1]   #Out = [-1,28,28,32]
conv1=conv_layer(ImageBatch, 1, 32, name='conv1')

#In = [-1,28,28,32]   #Out = [-1,14,14,32]
pooling1=pooling_layer(conv1, name='pooling1')

#In = [-1,14,14,32]   #Out = [-1,14,14,64]
conv2=conv_layer(pooling1, 32, 64, name='conv2')

#In = [-1,14,14,64]   #Out = [-1,7,7,64]
pooling2=pooling_layer(conv2, name='pooling2')

#In = [-1,7,7,64]   #Out = [-1,7*7*64]
flatImages=tf.reshape(pooling2, (-1,7*7*64))

#In = [-1,7*7*64]   #Out = [-1,1024]
dense1=dense_layer(flatImages, 7*7*64, 1024, name='dense1')

#In = [-1,1024]   #Out = [-1,10]
logits=dense_layer(dense1, 1024, 10, name='dense2')



with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=LabelBatch))
    
tf.summary.scalar('loss', loss)
     
with tf.name_scope('train'):
    train=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    
with tf.name_scope('accuracy'):
    accuracy=tf.reduce_mean((tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(LabelBatch,1)), tf.int32)))
    
tf.summary.scalar('accuracy', accuracy)

merger=tf.summary.merge_all()


epocs=100

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    writer=tf.summary.FileWriter('C:\Workspace\PythonProject\TensorFlow\TensorFlow_Learning\ConvNN\TensorBoard\summery2')
    writer.add_graph(sess.graph)

    for i in range(epocs):
          if i%5==0 :
              Merger, Loss, Accuracy = sess.run((merger, loss, accuracy))
              print('Loss {} Accuracy {}'.format(Loss, Accuracy))
              writer.add_summary(Merger,i)
          sess.run(train)
















   
    
   
