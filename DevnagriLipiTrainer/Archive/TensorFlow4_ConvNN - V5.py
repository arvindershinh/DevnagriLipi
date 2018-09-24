# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:26:40 2018

@author: Arvinder Shinh
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import os



imageFiles=os.listdir('image')

SerializedImgContainer=[]
LabelContainer=[]

for f in imageFiles:
   if f.endswith('.jpg'):
    fname, fext = os.path.splitext(f)
    label = 0 if fname == 'ka' else 1
    image=Image.open(os.path.join('image',f))
    image=image.resize((28,28))
    image=image.convert(mode='L')
    image=np.array(image).reshape((28,28,1))
    image=image.tostring()
    
    FloatList1=tf.train.FloatList(value=image) 
    
    SerializedImage=tf.train.Feature(float_list=FloatList1)
    
    Features_Map={'image': SerializedImage}
    Features=tf.train.Features(feature=Features_Map)
    Example=tf.train.Example(features=Features).SerializeToString()
    
    SerializedImgContainer.append(Example)
    LabelContainer.append(label)
    

Num_Exp=4
a=np.random.randint(0,9,Num_Exp)

labels=np.zeros((Num_Exp,10),dtype=np.int32)
for i in range(Num_Exp-1):
    labels[i,a[i]]=1
    

SerialImgPlaceHolder=tf.placeholder(dtype=tf.string, name='SerializedImages')
Feature_trans={'image': tf.FixedLenFeature(shape=(784), dtype=tf.float32)}
data=tf.parse_example(SerialImgPlaceHolder, Feature_trans)

x=tf.reshape(data['image'], shape=(-1,28,28,1), name='Images')

y=tf.placeholder(shape=(None,10),dtype=tf.float32, name='Labels')

data=tf.data.Dataset.from_tensor_slices({'x': x, 'y': y})
data=data.shuffle(100).repeat().batch(5)

iterator=data.make_initializable_iterator()

batch=iterator.get_next()
ImageBatch=batch['x']
LabelBatch=batch['y']



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

def ConvolutionNN(learning_rate, Num_ConvLayer, Num_DenseLayer, HyperParaStr, name='HyperParameters'):

 with tf.name_scope(name):   
  if Num_ConvLayer == 2 and Num_DenseLayer == 2:
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
     
  elif Num_ConvLayer == 1 and Num_DenseLayer == 2:
#In = [-1,28,28,1]   #Out = [-1,28,28,32]
     conv1=conv_layer(ImageBatch, 1, 32, name='conv1')

#In = [-1,28,28,32]   #Out = [-1,14,14,32]
     pooling1=pooling_layer(conv1, name='pooling1')

       
#In = [-1,14,14,32]   #Out = [-1,14*14*32]
     flatImages=tf.reshape(pooling1, (-1,14*14*32))

#In = [-1,14*14*32]   #Out = [-1,1024]
     dense1=dense_layer(flatImages, 14*14*32, 1024, name='dense1')

#In = [-1,1024]   #Out = [-1,10]
     logits=dense_layer(dense1, 1024, 10, name='dense2')
  
#  elif Num_ConvLayer == 2 and Num_DenseLayer == 1:
#  elif Num_ConvLayer == 1 and Num_DenseLayer == 2:
#  elif Num_ConvLayer == 1 and Num_DenseLayer == 1:

  with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=LabelBatch))
    
  tf.summary.scalar('loss', loss)
     
  with tf.name_scope('train'):
    train=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
  with tf.name_scope('accuracy'):
    accuracy=tf.reduce_mean((tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(LabelBatch,1)), tf.int32)))
    
  tf.summary.scalar('accuracy', accuracy)

  merger=tf.summary.merge_all()


  epocs=100
  path="C:/Workspace/PythonProject/TensorFlow/TensorFlow_Learning/ConvNN/TensorBoard/"+HyperParaStr

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={SerialImgPlaceHolder: SerializedImgContainer, y: labels})
    writer=tf.summary.FileWriter(path)
    writer.add_graph(sess.graph)

    print('Active Hyper Parameter'+HyperParaStr)
    for i in range(epocs):
          if i%5==0 :
              Merger, Loss, Accuracy = sess.run((merger, loss, accuracy), feed_dict={SerialImgPlaceHolder: SerializedImgContainer, y: labels})
              print('Loss {} Accuracy {}'.format(Loss, Accuracy))
              writer.add_summary(Merger,i)
          sess.run(train, feed_dict={SerialImgPlaceHolder: SerializedImgContainer, y: labels})




def HyperParameterStr(learning_rate,Num_ConvLayer,Num_DenseLayer):
    return "LR= lr_%.0E,ConvLayer=%s,DenseLayer=%s" % (learning_rate, Num_ConvLayer, Num_DenseLayer)


def main():
    learning_rates =[1e-4,1e-5]
    i=0
    Num_ConvLayers=[1,2]
    Num_DenseLayers=[2]
    
    for learning_rate in learning_rates:
        for Num_ConvLayer in Num_ConvLayers:
            for Num_DenseLayer in Num_DenseLayers:
                i=i+1
                HyperParaStr=HyperParameterStr(learning_rate,Num_ConvLayer,Num_DenseLayer)
                ConvolutionNN(learning_rate, Num_ConvLayer, Num_DenseLayer, HyperParaStr, 'HyperParameter%s' % (i))
    

if __name__ == '__main__':
    main()





   
    
   
