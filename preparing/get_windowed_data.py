#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


# In[4]:


raw_data = pd.read_csv('2015_labeled_and_time_normalize.csv').drop(['Unnamed: 0', 'Timestamp'], axis=1)


# In[5]:


raw_data.columns


# In[6]:


def get_new_headers(headers, step):
    result = []
    for i in range(step):
        for j in headers:
            result.append(j.strip() + '_'+ str(i))
    return result


# In[7]:


def prepare_frame(frame, new_headers, step=4):
    headers = frame.columns
    d = preprocessing.normalize(frame)
    scaled_frame = pd.DataFrame(d, columns=headers)
    r = []
    labels = []
    for i in range(0, int(len(scaled_frame)/4)*4-4):
        t = []
        attack = False 
        for j in range(step):
            #print(scaled_frame.values[i+j][-1])
            if scaled_frame.values[i+j][-1] >  0:
                attack = True
            #except:
               # print(i, j)
               # raise
            t.append(scaled_frame.values[i+j][:-1])
        r.append(np.concatenate(t))
        if attack:
            labels.append(1)
        else:
            labels.append(0)
    return pd.DataFrame(r, columns=new_headers), labels


# In[8]:


new_headers = get_new_headers(raw_data.drop(['Normal/Attack'], axis=1).columns, 4)


# In[9]:


train = prepare_frame(raw_data, new_headers)


# In[12]:

train[0].to_csv('data.csv', columns=new_headers)
import pickle


# In[13]:


#with open('data.pickle', 'wb') as f:
 #   pickle.dump(train[0], f)


# In[14]:


with open('lablels.pickle', 'wb') as f:
    pickle.dump(train[1], f)

