#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[2]:


get_ipython().run_line_magic('cd', "'/content/gdrive/My Drive/Kaggle'")


# In[3]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
import keras.utils
from keras import utils as np_utils


# In[4]:


import numpy as np
CATEGORIES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
 'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',]

def image(path):
    img = cv2.imread(path)
    new_arr = cv2.resize(img,(100, 100))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    return new_arr

model = keras.models.load_model('Leaf Deases(96,88).h5')


# In[6]:


prediction = model.predict(image('/content/LateBlightFig3new.jpg'))

print(CATEGORIES[prediction.argmax()])


# In[ ]:





# # New Section
