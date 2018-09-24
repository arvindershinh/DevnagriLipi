# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:13:28 2018

@author: Arvinder Shinh
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def DevnagriDataVerifier(ImageBatch, LabelBatch):
    DevnagriAlphabets=['ba','g', 'ka', 'kha', 'la', 'ma', 'pa', 'ra', 'ta', 'tha']
    Label=DevnagriAlphabets[np.argmax(LabelBatch)]
    Image1=ImageBatch.reshape((28,28))
    Image2=Image1.astype(int)
    Image3=Image.fromarray(Image2)
    print(Label)
    fig=plt.figure()
    ax=fig.subplots(1,1)
    ax.imshow(Image3)
    plt.show()
    