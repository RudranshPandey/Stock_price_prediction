#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[2]:


start ='2010-01-01'
end = '2019-12-31'

df=data.DataReader('AAPL','yahoo', start, end )
df.head()


# In[3]:


df=df.reset_index()
df.head()


# In[4]:


df=df.drop(['Date','Adj Close'],axis=1)
df.head()


# In[5]:


plt.plot(df.Close)


# In[6]:


df


# In[7]:


ma100=df.Close.rolling(100).mean()
ma100


# In[8]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')


# In[9]:


ma200=df.Close.rolling(200).mean()
ma200


# In[10]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# In[11]:


df.shape


# In[12]:


# Spliting data into two DataFrames ie. Testing and Traning 

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)
                             


# In[13]:


data_training.head()


# In[14]:


data_testing.head()


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[16]:


data_training_array = scaler.fit_transform(data_training)
data_training_array


# In[18]:


x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)


# In[19]:


# ML model


# In[20]:


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[21]:


model = Sequential()
model.add(LSTM(units = 50, activation ='relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation ='relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation ='relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation ='relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


# In[22]:


model.summary()


# In[26]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 50)


# In[40]:


model.save('Ml_model_LSTM')


# In[41]:


data_training.tail(100)


# In[42]:


data_testing.head()


# In[43]:


# we need to append the values from tail to the values of head only then we will be able to predict the values based on the head
#position


# In[44]:


past_100_days = data_training.tail(100)


# In[45]:


final_df = past_100_days.append(data_testing, ignore_index = True)


# In[46]:


final_df.head()


# In[47]:


input_data = scaler.fit_transform(final_df)
input_data


# In[48]:


input_data.shape


# In[50]:


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)


# In[51]:


print(x_test.shape)
print(y_test.shape)


# In[52]:


#making Predictions

y_predicted = model.predict(x_test)


# In[53]:


y_predicted.shape


# In[54]:


y_test


# In[55]:


y_predicted


# In[56]:


scaler.scale_


# In[57]:


scale_factor = 1/0.02099517
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor 


# In[58]:


plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[59]:


y_predicted.shape


# In[60]:


y_test.shape


# In[ ]:




