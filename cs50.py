
# coding: utf-8

# In[1]:

get_ipython().magic('lsmagic')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

x_train = np.array([[1,1],[2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6]])
y_train = ['red','red','red','blue','blue','blue']


# In[4]:

print(x_train[5,0])
print(x_train[5,1])
print(x_train[:,0])
print(x_train[:,1])


# In[5]:

x_test = np.array([3,4])


# In[6]:

plt.figure()
plt.scatter(x_train[:,0],x_train[:,1],s=170,color=y_train[:])
plt.scatter(x_test[0],x_test[1],s=170,color='green')
plt.show()


# In[7]:

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))


# In[8]:

num = len(x_train)
distance = np.zeros(num)
for i in range(num):
    distance[i] = dist(x_train[i],x_test)
print(distance)


# In[9]:

min_index = np.argmin(distance)
print(y_train[min_index])


# In[10]:

from sklearn import datasets


# In[11]:

digits = datasets.load_digits()
print(digits.images[0])


# In[12]:

plt.figure()
plt.imshow(digits.images[0],cmap = plt.cm.gray_r, interpolation = 'nearest')
plt.show()


# In[13]:

print(digits.target[0])


# In[14]:

x_train = digits.data[0:10]
y_train = digits.target[0:10]


# In[15]:

x_test = digits.data[345]


# In[16]:

plt.figure()
plt.imshow(digits.images[345], cmap = plt.cm.gray_r, interpolation ='nearest')
plt.show()


# In[17]:

num = len(x_train)
distance = np.zeros(num)
for i in range(num):
    distance[i] = dist(x_train[i],x_test)
min_index = np.argmin(distance)
print(y_train[min_index])


# In[18]:

num = len(x_train)
distance = np.zeros(num)
for i in range(num):
    distance[i] = dist(x_train[i],x_test)
min_index = np.argmin(distance)
print(y_train[min_index])


# In[19]:

num  = len(x_train)
no_errors = 0
distance = np.zeros(num)
for j in range(1697,1797):
    x_test = digits.data[j]
    for i in range(num):
        distance[i] = dist(x_train[i],x_test)
    min_index = np.argmin(distance)
    if y_train[min_index] != digits.target[j]:
        no_errors += 1
print(no_errors)


# In[20]:

x_train = digits.data[0:1000]
y_train = digits.target[0:1000]
num  = len(x_train)
no_errors = 0
distance = np.zeros(num)
for j in range(1697,1797):
    x_test = digits.data[j]
    for i in range(num):
        distance[i] = dist(x_train[i],x_test)
    min_index = np.argmin(distance)
    if y_train[min_index] != digits.target[j]:
        no_errors += 1
print(no_errors)


# In[21]:

X = np.array([[1,1],[2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6],[8,8]])


# In[22]:

plt.figure()
plt.scatter(X[:,0],X[:,1],s=170,color='black')
plt.show()


# In[23]:

from sklearn.cluster import KMeans


# In[24]:

k=2
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


# In[25]:

colors = ['r.','g.']
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]],markersize = 30)
plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=300,linewidths=5)
plt.show()


# In[26]:

k=3
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
colors = ['r.','g.','y.']
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]],markersize = 30)
plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=300,linewidths=5)
plt.show()


# In[27]:

k=7
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
colors = ['r.','g.','y.','c.','b.','k.','m.']
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]],markersize = 30)
plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=300,linewidths=5)
plt.show()


# In[28]:

corpus = ['I love CS50. Staff is awesome, awesome, awesome!',
          'I have a dog and a cat.',
          'Best of CS50? Staff. And cakes. Ok, CS50 staff.',
          'My dog keeps chasing my cat. Dogs!']


# In[29]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words = 'english')
z = count_vect.fit_transform(corpus)
z.todense()


# In[30]:

vocab = count_vect.get_feature_names()
print(vocab)


# In[31]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(corpus)
X.todense()


# In[32]:

k=2
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X)
model = KMeans(n_clusters=k)
model.fit(X)


# In[33]:

print("Top terms per cluster:\n")
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %i:" % i, end='')
    for ind in order_centroids[i,:3]:
        print(' %s,' % terms[ind], end='')
    print("")


# In[36]:

import pandas as pd
from io import StringIO
import requests

act = requests.get('https://docs.google.com/spreadsheets/d/1udJ4nd9EKlX_awB90JCbKaStuYh6aVjh1X6j8iBUXIU/export?format=csv')
dataact = act.content.decode('utf-8')
frame = pd.read_csv(StringIO(dataact))
print(frame)


# In[37]:

corpus = []
for i in range(0, frame["Synopsis"].size):
    corpus.append(frame["Synopsis"][i])


# In[39]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 0.2)
X = vectorizer.fit_transform(corpus)


# In[40]:

k = 2 
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X)
model = KMeans(n_clusters = k)
model.fit(X);


# In[41]:

no_words = 4 
order_centroids = model.cluster_centers_.argsort()[:, ::-1] 
terms = vectorizer.get_feature_names()
labels = model.labels_ 

print("Top terms per cluster:\n")
for i in range(k):
    
    print("Cluster %d movies:" % i, end='')
    for title in frame["Title"][labels == i]:
        print(' %s,' % title, end='')
    print() #add a whitespace

    print("Cluster %d words:" % i, end='') 
    for ind in order_centroids[i, :no_words]:
        print (' %s' % terms[ind], end=','),
    print()
    print()

