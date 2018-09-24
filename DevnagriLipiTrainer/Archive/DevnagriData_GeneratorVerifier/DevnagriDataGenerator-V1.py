# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 20:18:40 2018

@author: Arvinder Shinh
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import os
from math import pi


def colorflipper(image):
     image=np.array(image)
     imageBoolMoreWhite = image > 144.5
     imageBoolMoreBlack = image < 144.5
     image[imageBoolMoreWhite] = 0
     image[imageBoolMoreBlack] = 255
     return Image.fromarray(image)
     
 
def Rotation(image, angle):
    if angle <= pi/8 and angle >= -pi/8:
     return image.rotate(180/pi*angle,)
 

def Translation(image, dx, dy):
    
   image=np.array(image)
   xDim, yDim=image.shape
   
    # dx->Column dy->Row
   if dx == True:
       dx=int(xDim/10)
   elif dx == False:
       dx=-int(xDim/10)
       
   if dy == True:
       dy=int(xDim/10)
   elif dy == False:
       dy=-int(xDim/10)
        
    # left-top done
   if dx < 0 and dy > 0:
    image_dx=image[:,:-dx]
    image_cropped1=image[:,-dx:]
    image_dy=image_cropped1[:dy,:]
    image_cropped2=image[dy:,-dx:]
    image_merged1=np.vstack((image_cropped2, image_dy))
    image_transported=np.hstack((image_merged1, image_dx))
    
    # left-down done
   if dx < 0 and dy < 0:
    image_dx=image[:,:-dx]
    image_cropped1=image[:,-dx:]
    image_dy=image_cropped1[dy:,:]
    image_cropped2=image[:dy,-dx:]
    image_merged1=np.vstack((image_dy, image_cropped2))
    image_transported=np.hstack((image_merged1, image_dx))
    
    #right-top done
   if dx > 0 and dy > 0:
    image_dx=image[:,-dx:]
    image_cropped1=image[:,:-dx]
    image_dy=image_cropped1[:dy,:]
    image_cropped2=image[dy:,:-dx]
    image_merged1=np.vstack((image_cropped2, image_dy))
    image_transported=np.hstack((image_dx, image_merged1))
    
    #right-down    
   if dx > 0 and dy < 0:
    image_dx=image[:,-dx:]
    image_cropped1=image[:,:-dx]
    image_dy=image_cropped1[dy:,:]
    image_cropped2=image[:dy,:-dx]
    image_merged1=np.vstack((image_dy, image_cropped2))
    image_transported=np.hstack((image_dx, image_merged1))
    
   return Image.fromarray(image_transported)


def DataGeneratingFunction(filePath, rootFolder):
    if filePath.endswith('.png'):
      image=Image.open(filePath)
      image=image.convert(mode='L')
      image=colorflipper(image)   
      filename=filePath.split('\\')[-1]
      fname, fext = os.path.splitext(filename)
      angles=np.linspace(-pi/8, pi/8, 201)
      n=angles.shape[0]
      i=0
      dx, dy = 50, 50
      for i in range(n):
        m=0  
        Rotated_Image0=Rotation(image, angles[i])
        Rotated_Image1=colorflipper(Rotated_Image0) 
        Rotated_Image2=Rotated_Image1.resize((28,28))
        Rotated_Image2.save(os.path.join(rootFolder, fname+str(i)+str(m)+fext))
        for j in [1, -1]:
            for k in [1, -1]:
                m=m+1
                Translated_Image=Translation(Rotated_Image1, j*dx, k*dy)
                Translated_Image=Translated_Image.resize((28,28))
                Translated_Image.save(os.path.join(rootFolder, fname+str(i)+str(m)+fext))
                

def DevnagriData(DevnagriImageFolder='DevnagriImage'):
    if not os.path.isdir(DevnagriImageFolder):
        os.mkdir(DevnagriImageFolder)
        DevnagriFolder=os.listdir('DevnagriImageData')
        for fo in DevnagriFolder:
           HindiLetterFolderPath=os.path.join('DevnagriImageData',fo)   
           HindiLetterFolder=os.listdir(HindiLetterFolderPath)
           for fi in HindiLetterFolder:
               DataGeneratingFunction(os.path.join(HindiLetterFolderPath, fi), "DevnagriImage")
    
    SerializedImgContainer=[]
    LabelContainer=[]
    DevnagriAlphabets=['ba','g', 'ka', 'kha', 'la', 'ma', 'pa', 'ra', 'ta', 'tha']    
    DevnagriImage=os.listdir(DevnagriImageFolder)
    
    for f in DevnagriImage:
      if f.endswith('.png'):
        fname, fext = os.path.splitext(f)
        label = DevnagriAlphabets.index(fname.split('_')[0])
        
        image=Image.open(os.path.join(DevnagriImageFolder,f))
        image=np.array(image).reshape((28,28,1))
        image=image.tostring()
        
        FloatList1=tf.train.FloatList(value=image) 
        
        SerializedImage=tf.train.Feature(float_list=FloatList1)
        
        Features_Map={'image': SerializedImage}
        Features=tf.train.Features(feature=Features_Map)
        Example=tf.train.Example(features=Features).SerializeToString()
        
        SerializedImgContainer.append(Example)
        LabelContainer.append(label)
        
    labelIndex=np.array(LabelContainer)
    Num_Exp = labelIndex.shape[0]
    
    # Converting to Hot Vector
    Labels=np.zeros((Num_Exp,10),dtype=np.int32)
    for i in range(Num_Exp-1):
        Labels[i,labelIndex[i]]=1
                
    return SerializedImgContainer, Labels       
