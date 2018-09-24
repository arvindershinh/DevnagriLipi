# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:26:40 2018

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

Predict_Inputs=tf.reshape(data['image'], shape=(-1,28,28,1), name='Predict_Inputs')

Train_Outputs=tf.placeholder(shape=(None,10),dtype=tf.float32, name='Labels')

data=tf.data.Dataset.from_tensor_slices({'x': Predict_Inputs, 'y': Train_Outputs})
data=data.shuffle(46500).repeat().batch(5)

iterator=data.make_initializable_iterator()

batch=iterator.get_next()
ImageBatch=batch['x']
LabelBatch=batch['y']


'''Convolution Layer'''
def conv_layer(inputs, In, Out, name='conv'):
    with tf.name_scope(name):
      with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w=tf.get_variable('w', initializer=tf.random_normal((5,5,In,Out)), dtype=tf.float32)
        b=tf.get_variable('b', initializer=tf.zeros((Out)), dtype=tf.float32)
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
      with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  
        w=tf.get_variable('w', initializer=tf.random_normal((In,Out)), dtype=tf.float32)
        b=tf.get_variable('b', initializer=tf.zeros((Out)), dtype=tf.float32)
        dense=tf.matmul(inputs,w)
        activation=tf.nn.relu(dense+b)
        tf.summary.histogram(name+'_w_kernal', w)
        tf.summary.histogram(name+'_b_kernal', b)
        tf.summary.histogram(name+'_activation', activation)
        return activation

def ConvolutionNN(learning_rate, Num_ConvLayer, Num_DenseLayer, HyperParaStr, i='0'):
 name='Devnagri'+i
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
     
#In = [-1,1024]     #Out = [-1,1024]     
     dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=True, name='dropout1')

#In = [-1,1024]   #Out = [-1,10]
     logits=dense_layer(dropout1, 1024, 10, name='dense2')
     
     
  elif Num_ConvLayer == 1 and Num_DenseLayer == 2:
#In = [-1,28,28,1]   #Out = [-1,28,28,32]
     conv1=conv_layer(ImageBatch, 1, 32, name='conv1')

#In = [-1,28,28,32]   #Out = [-1,14,14,32]
     pooling1=pooling_layer(conv1, name='pooling1')
     
#In = [-1,14,14,32]   #Out = [-1,14*14*32]
     flatImages=tf.reshape(pooling1, (-1,14*14*32))

#In = [-1,14*14*32]   #Out = [-1,1024]
     dense1=dense_layer(flatImages, 14*14*32, 1024, name='dense1')
     
#In = [-1,1024]     #Out = [-1,1024]     
     dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=True, name='dropout1')

#In = [-1,1024]   #Out = [-1,10]
     logits=dense_layer(dropout1, 1024, 10, name='dense2')
     
  else:
        print('Number of Convolution layers allowed are either 1 or 2 and Dense Layers allowed are only 2')
        return
  
     
  Predict_Outputs = tf.nn.softmax(logits, 1, name='Predict_Outputs')
  Classify_Output_Scores, indices = tf.reduce_max(Predict_Outputs, 1, name='values'), tf.argmax(Predict_Outputs, 1, name='indices')
     
  table=tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(10)]))
  Classify_Output_Classes=table.lookup(tf.to_int64(indices))

     
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=LabelBatch))
    
  tf.summary.scalar('loss', loss)
     
  with tf.name_scope('train'):
    train=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
  with tf.name_scope('accuracy'):
    accuracy=tf.reduce_mean((tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(LabelBatch,1)), tf.int32)))
    
  tf.summary.scalar('accuracy', accuracy)

  merger=tf.summary.merge_all()

  TensorBoardPath="Output/TensorBoard/"+HyperParaStr
  
  saver = tf.train.Saver()
  VariableCkptPath='C:/Workspace/PythonProject/TensorFlow/Models/DevnagriLipi/Output/VariableCkpt/'+HyperParaStr+'/'+name+'.ckpt'

  epocs=930
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={Classfy_Inputs: SerializedImgContainer, Train_Outputs: Labels})
    
    writer=tf.summary.FileWriter(TensorBoardPath)
    writer.add_graph(sess.graph)

    for i in range(epocs):
      if i%5==0 :
          Merger, Loss, Accuracy, imageBatch, labelBatch = sess.run((merger, loss, accuracy, ImageBatch, LabelBatch), feed_dict={Classfy_Inputs: SerializedImgContainer, Train_Outputs: Labels})
          print('Loss {} Accuracy {}'.format(Loss, Accuracy))
          writer.add_summary(Merger,i)
          
      '''Data Verification'''  
      if i%50==0:
          DevnagriLipiDataVerifier.DevnagriDataVerifier(imageBatch[0], labelBatch[0])
 
      sess.run(train, feed_dict={Classfy_Inputs: SerializedImgContainer, Train_Outputs: Labels})

    '''Save Model Variables'''      
    saver.save(sess, VariableCkptPath)
          
          
    '''Serving'''      
    Classify_Inputs_proto=sm.utils.build_tensor_info(Classfy_Inputs)
    Classify_Output_Classes_proto=sm.utils.build_tensor_info(Classify_Output_Classes)
    Classify_Output_Scores_proto=sm.utils.build_tensor_info(Classify_Output_Scores)
     
    ClassifySignatureDef=(sm.signature_def_utils.build_signature_def(
             inputs={sm.signature_constants.CLASSIFY_INPUTS: Classify_Inputs_proto},
             outputs={sm.signature_constants.CLASSIFY_OUTPUT_CLASSES: Classify_Output_Classes_proto,
                      sm.signature_constants.CLASSIFY_OUTPUT_SCORES: Classify_Output_Scores_proto},
             method_name=sm.signature_constants.CLASSIFY_METHOD_NAME))
     
    Predict_Inputs_proto=sm.utils.build_tensor_info(Predict_Inputs)
    Predict_Outputs_proto=sm.utils.build_tensor_info(Predict_Outputs)
     
    PredictSignatureDef=(sm.signature_def_utils.build_signature_def(
             inputs={'image': Predict_Inputs_proto},
             outputs={'scores': Predict_Outputs_proto},
             method_name=sm.signature_constants.PREDICT_METHOD_NAME))
    
    ServingPath='Output/ServingModel/'+HyperParaStr+'/DevnagriLipiPredictor/1'
    SavedModel=sm.builder.SavedModelBuilder(export_dir=ServingPath)
    SavedModel.add_meta_graph_and_variables(sess, 
                                             [sm.tag_constants.SERVING],
                                            signature_def_map={'serving': PredictSignatureDef,
                                                               sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: ClassifySignatureDef}, 
                                            strip_default_attrs=True)
    
    SavedModel.save()


def HyperParameterStr(learning_rate,Num_ConvLayer,Num_DenseLayer):
     learning_rate_str='%.0E'%(learning_rate)
     learning_rate_digits=learning_rate_str.split('-')[1]
#    return "LRlr_%.0E,ConvLayer%s,DenseLayer%s" % (learning_rate, Num_ConvLayer, Num_DenseLayer)   # Causing issue in opening tensor board
     return "Devnagri_{}_{}_{}".format(learning_rate_digits, Num_ConvLayer, Num_DenseLayer)


def main():
    learning_rates =[1e-4]
    i=0
    Num_ConvLayers=[2]
    Num_DenseLayers=[2]
    
    for learning_rate in learning_rates:
        for Num_ConvLayer in Num_ConvLayers:
            for Num_DenseLayer in Num_DenseLayers:
                i=i+1
                HyperParaStr=HyperParameterStr(learning_rate,Num_ConvLayer,Num_DenseLayer)
                print('Active Hyper Parameter '+HyperParaStr+' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                ConvolutionNN(learning_rate, Num_ConvLayer, Num_DenseLayer, HyperParaStr, str(i))
    

if __name__ == '__main__':
    main()





   
    
   
