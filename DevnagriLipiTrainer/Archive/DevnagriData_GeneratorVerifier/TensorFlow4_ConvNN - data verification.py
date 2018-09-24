# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:26:40 2018

@author: Arvinder Shinh
"""
import tensorflow as tf
import DevnagriDataGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


SerializedImgContainer, Labels = DevnagriDataGenerator.DevnagriData()

Classfy_Inputs=tf.placeholder(dtype=tf.string, name='Classfy_Inputs')

Feature_trans={'image': tf.FixedLenFeature(shape=(784), dtype=tf.float32)}
data=tf.parse_example(Classfy_Inputs, Feature_trans)

Predict_Inputs=tf.reshape(data['image'], shape=(-1,28,28,1), name='Predict_Inputs')

Train_Outputs=tf.placeholder(shape=(None,10),dtype=tf.float32, name='Labels')

data=tf.data.Dataset.from_tensor_slices({'x': Predict_Inputs, 'y': Train_Outputs})
data=data.shuffle(301500).repeat().batch(5)

iterator=data.make_initializable_iterator()

batch=iterator.get_next()
ImageBatch=batch['x']
LabelBatch=batch['y']

ImageBatch_0=ImageBatch[0]
LabelBatch_0=LabelBatch[0]


epocs=931

def main():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={Classfy_Inputs: SerializedImgContainer, Train_Outputs: Labels})
    #    ['ba','g', 'ka', 'kha', 'la', 'ma', 'pa', 'ra', 'ta', 'tha'] 
    for i in range(epocs):
       if i%100 == 0:
        TestImage1, TestLabel = sess.run((ImageBatch_0, LabelBatch_0), feed_dict={Classfy_Inputs: SerializedImgContainer, Train_Outputs: Labels})
        TestImage6=TestImage1.reshape((28,28))
        TestImage7=TestImage6.astype(int)
        TestImage3=Image.fromarray(TestImage7)
        print('++++++++++++++++++++++++++++++++++++++')
        print(TestLabel)
        fig=plt.figure()
        ax=fig.subplots(1,1)
        ax.imshow(TestImage3)
        plt.show()   
              


if __name__ == '__main__':
    main()





   
    
   
