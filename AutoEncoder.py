# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:10:15 2021

@author: icek1
"""
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data=pd.read_csv("mnist_train.csv")
sample=data[list(data.columns)[1:]].values   #取数据
label=data.iloc[:,0]            #取第一列，也是标签值(数字大小)

data=pd.read_csv("mnist_test_10.csv") #验证数据
test=data[list(data.columns)[1:]].values

sample=sample/255

for x in range(0,len(sample)):
    
    sample[x] = sample[x].reshape(-1, 784)

for x in range(0,len(test)):
    test[x] = test[x].reshape(-1, 784)


for i in range(0,9):
    plt.imshow(test[i].reshape(28,28))
    plt.show()
# reshape the inputs



input_layer=Input(shape=(784,))
encode_layer1=Dense(1500,activation='relu')(input_layer) #全连接层
encode_layer2=Dense(1000,activation='relu')(encode_layer1)
encode_layer3=Dense(500,activation='relu')(encode_layer1)

latent_view   = Dense(10, activation='sigmoid')(encode_layer3)

# 解码层

decode_layer1 = Dense(500,activation='relu')(latent_view)
decode_layer2 = Dense(1000,activation='relu')(decode_layer1)
decode_layer3 = Dense(1500,activation='relu')(decode_layer2)


out_layer=Dense(784)(decode_layer3)

mod=Model(input_layer,out_layer)
mod.summary()
mod.compile(optimizer="adam",loss="mse")

EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=1,mode='auto')
mod.fit(sample,sample,epochs=20,batch_size=2048,validation_data=(test,test))


pred = mod.predict(test)

for i in range(9):
    plt.imshow(pred[i].reshape(28,28))
    plt.show()
