#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Predicting Diabetes Exsistence


# In[11]:


#Loading the dataset
import numpy as np
data = np.loadtxt("C:/Users/91799/Downloads/diabetes.csv", delimiter=',', skiprows=1)
data


# In[12]:


#Train the Model
X=data[:,0:8]
Y=data[:,8]


# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[19]:


model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[20]:


model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])


# In[21]:


model.fit(X,Y,epochs = 150,batch_size = 10)


# In[22]:


_,accuracy=model.evaluate(X,Y)
print(accuracy*100)


# In[24]:


predictions = model.predict(X)


# In[27]:


for i in range(5):
    print(X[i].tolist(),"Predicted-->",predictions[i],"Actual -->",Y[i])


# In[ ]:




