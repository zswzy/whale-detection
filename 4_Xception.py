import pandas as pd
import numpy as np
import os
from skimage import io
from skimage import img_as_ubyte
from skimage import img_as_float64
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

imgPath = '/Users/Zeyuan/Documents/Kaggle/humpback-whale-identification/all_reshape2'
imglist = os.listdir(imgPath)
imglist = imglist[0:1001]
print('imgs number:',len(imglist))
print('image shape:',io.imread('all_reshape2/%s'%imglist[0]).shape)

print('reading images')
i = 1
pixels = []
for img_name in sorted(imglist)[1:]: #需要按文件名称顺序排好 排除第一个元素checkpoint  
    
    img_gray = io.imread('all_reshape2/%s'%img_name, as_gray=True)
    img_gray_float64 = img_as_float64(img_gray)
    img_gray_1D = np.reshape(img_gray_float64,(1,128*248))
    pixels.append(img_gray_1D[0])
    print('\rprocessing %d images(%.2f%%), %s images to go.'
          %(i, (i/len(imglist)*100), (len(imglist)-i)),end='')
    i+=1
print('\nreading finish')

print('\rConverting to numpy array...',end='')
pixels = np.array(pixels)
print('\rConverting to numpy array...finish')

print('\rAdding labels...',end='')
train_df = pd.read_csv('train.csv')
raw = np.concatenate((train_df[['Id']][0:1000].values,pixels),axis = 1)
print('\rAdding labels...finish')

# %%
img_rows, img_cols = 128, 248 
num_classes = 5005

def data_prep(raw): 
    #raw是ndarray，每一行是一张图片的一维展开，第一列是标签
    #输出out_x, out_y
    le = LabelEncoder()
    le.fit(raw[:,0])
    label = le.transform(raw[:,0])
    out_y = keras.utils.to_categorical(label, 5005)
    
    num_images = raw.shape[0]
    x = raw[:,1:].astype(np.float32)
    out_x = np.reshape(x, (num_images, img_rows, img_cols,1))
    out_x = np.concatenate((out_x, out_x, out_x), -1)#强制转换为RGB
    
    return out_x, out_y

print('\rdata preparing...',end='')
raw_data = raw
x, y = data_prep(raw_data)
print('\rdata preparing...finish')
print('x shape:',x.shape,'\ny shape:',y.shape)
# %%
print('\rsaving array...',end='')
np.save('raw(500).npy',raw)
print('\rsaving array...finish')

# %%
print('\rCreating model...',end='')

model = Sequential()
model.add(keras.applications.xception.Xception(include_top=False, pooling='max',
                                weights='imagenet'))
model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable = False

model.summary()
model.compile(optimizer='sgd', 
                     loss=keras.losses.categorical_crossentropy, 
                     metrics=['accuracy'])
print('Training...')
model.fit(x, y,
          batch_size=50,
          epochs=2,
          validation_split = 0.2)
print('Training finish')
