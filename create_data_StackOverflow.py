from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/stack-overflow-data.csv")
    
InputData = data['post'].tolist()
OutputData = data['tags'].tolist()

data_df = pd.DataFrame({'label':OutputData, 'text':InputData})

label_counts = pd.DataFrame(data_df['label'].value_counts()).transpose()
l_c = list(label_counts)

data_text=[]
data_label=[]
for i in l_c:
  if int(label_counts[i]) >= 100:
    d = data_df[data_df['label'] == i].text.tolist()
    for j in d:
      data_text.append(j)
      data_label.append(i)
  else:
    pass

data_df = pd.DataFrame({'label':data_label, 'text':data_text})

text = data_df.text.tolist()
label = data_df.label.tolist()

train_data, test_data, train_labels, test_labels = train_test_split(text, label, test_size=0.2) 

train_df = pd.DataFrame({'text':train_data,'label':train_labels})
test_df = pd.DataFrame({'text':test_data,'label':test_labels})

train_df.to_csv('data/train.csv', sep=',', index=False, header=True)
test_df.to_csv('data/test.csv', sep=',', index=False, header=True)

plt.figure(figsize = (11,5))
train_df['label'].value_counts().plot.bar()
plt.title('Label distribution for the Training Data')
plt.show()

plt.figure(figsize = (11,5))
test_df['label'].value_counts().plot.bar()
plt.title('Label distribution for the Testing Data')
plt.show()
