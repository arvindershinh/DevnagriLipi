# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:20:46 2018

@author: Arvinder Shinh
"""

import tensorflow as tf
from tensorflow import saved_model as sm
import DevnagriLipiDataGenerator
import DevnagriLipiDataVerifier


SerializedImgContainer, Labels = DevnagriLipiDataGenerator.DevnagriData()

Classfy_Inputs=tf.placeholder(dtype=tf.string, name='Classfy_Inputs')

Feature_trans={'image': tf.FixedLenFeature(shape=(784), dtype=tf.float32)}
data=tf.parse_example(Classfy_Inputs, Feature_trans)

Predict_Inputs=tf.reshape(data['image'], shape=(-1,28,28), name='Predict_Inputs')

Train_Outputs=tf.placeholder(shape=(None,10),dtype=tf.float32, name='Labels')

with tf.Session() as sess:
 imageBatch, labelBatch = sess.run(( Predict_Inputs, Train_Outputs), feed_dict={Classfy_Inputs: SerializedImgContainer, Train_Outputs: Labels})

print(imageBatch.shape, labelBatch.shape)
for i in range(imageBatch.shape[0]):
      DevnagriLipiDataVerifier.DevnagriDataVerifier(imageBatch[i], labelBatch[i])