#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import string
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from sklearn.ensemble import RandomForestClassifier

get_ipython().system('pip install --upgrade gensim')
# import gensim 
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
from gensim import models

get_ipython().system('pip install autocorrect')


# In[2]:


data = pd.read_csv("Reviews.csv", engine="python",error_bad_lines=False)
data = data[:10000]


# #### 資料前處理

# 取score、text特定欄位

# In[3]:


df = data.loc[ : ,['Score','Text']]
df.head()


# 轉score特徵，大於4者為1，反之為0
# 

# In[4]:


for index in df.index:
    if df.loc[index, 'Score'] >=4 :
        df.loc[index,'Score'] = 1
    else:
        df.loc[index,'Score'] =0

df.head()


# ### Text preprocessing

# 將切割的words重組回sentence

# In[5]:


def append_words(word_set):
    for text in word_set.values:
        sequ =''
        for word in text:
             sequ = sequ + ' ' + word
    return sequ


# **處理html網址**

# In[6]:


def dealt_html(sentence):
    cleanr = re.compile('<.*?>')
    new_s = re.sub(cleanr, ' ', sentence) 

    return new_s


# 去除標點符號

# In[7]:


def remove_punctuation(text):
    translator = str.maketrans('','', string.punctuation)
    return text.translate(translator)


# 去除數字

# In[8]:


def remove_numbers(text):
    new_text = re.sub(r'\d+', '', text)
    return new_text


# 處理一個詞中重複出現的字母（如：loooove->love）
# 
# 
# 處理正確拼寫（如：aple->apple）
# 
# 
# 
# 

# In[9]:


from nltk.metrics.distance import jaccard_distance
from scipy.spatial.distance import jaccard
import itertools
from autocorrect import Speller
from nltk.util import ngrams
nltk.download('words')
from nltk.corpus import words


def remove_repeated(text):
    new_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    return new_text

def correct_spell(text):
    spell = Speller(lang='en')
    corr_text=spell(text)
    return corr_text


# 將文字全部轉為小寫

# In[10]:


def convert_lowercase(text):
    new_text = text.lower()
    return new_text


# In[11]:


x_data = pd.DataFrame(columns=['N_Text'],index=df.index)
for index in df.index:
    x_data.N_Text[index] = remove_numbers(str(df.Text[index])) #去除數字
    x_data.N_Text[index] = remove_repeated(str(x_data.N_Text[index])) #去除詞中的重複字母
    x_data.N_Text[index] = dealt_html(str(x_data.N_Text[index])) # 處理html格式文字
  # x_data.N_Text[index] = correct_spell(str(x_data.N_Text[index])) #正確拼寫,因跑不出來故省略
    x_data.N_Text[index] = convert_lowercase(str(x_data.N_Text[index])) #轉成英文小寫
    x_data.N_Text[index] = remove_punctuation(str(x_data.N_Text[index])) #去除標點符號
    
x_data.head()


# 作名詞、動詞變化的還原（e.g.books=book）

# In[12]:


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return lemmas


# In[13]:


for index in x_data.index:
    t = pd.DataFrame(lemmatize_word(str(x_data.N_Text[index])))
    split_text = pd.DataFrame(t.values.T) 
    x_data.N_Text[index] = append_words(split_text)

x_data.head()


# 去除stop words

# In[14]:


nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import stopwords

stop_words = set(stopwords.words("english"))


# In[15]:


def remove_stopwords(text):
    stop_words = list(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


# In[16]:


# 去除stop words
for index in x_data.index:
    t = pd.DataFrame(remove_stopwords(remove_punctuation(str(x_data.N_Text[index]))))
    split_text = pd.DataFrame(t.values.T) 
    x_data.N_Text[index] = append_words(split_text)
x_data.head()


# #### 文字探勘前處理

# 1.tf-idf  

# In[17]:


x_tf = pd.DataFrame()
tfidfvectorizer = TfidfVectorizer(smooth_idf = True ,analyzer='word',stop_words= 'english')
tfidf_wm = tfidfvectorizer.fit_transform(x_data.N_Text)
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
x_tf = pd.concat([df.Score,df_tfidfvect.reindex(df.index)],axis =1)
# print(df_tfidfvect)
x_tf.head()


# 2.word2vec

# In[18]:


import gensim 
x_word2vec = x_data
x_word2vec['N_text_clean'] = x_word2vec['N_Text'].apply(lambda x: gensim.utils.simple_preprocess(x))
x_word2vec = pd.concat([df.Score,x_word2vec.N_text_clean.reindex(df.index)],axis=1)
x_word2vec.head()


# In[19]:


def avg_vector (X_train, X_test):
    num_vector = 100
    w2v_model = gensim.models.Word2Vec(X_train,window=5, min_count=2,vector_size = num_vector)
    words = set(w2v_model.wv.index_to_key) 
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) 
                         for ls in X_train]) 
    X_test_vect = np.array( [np.array([w2v_model.wv[i] for i in ls if i in words]) 
                         for ls in X_test])
    
    # 通過平均句子中包含的單詞的單詞向量來計算句子向量
    X_train_vect_avg = [] 
    for v in X_train_vect: 
        if v.size: 
            X_train_vect_avg.append(v.mean(axis=0)) 
        else: 
            X_train_vect_avg.append(np.zeros(100, dtype=float)) 

    X_test_vect_avg = [] 
    for v in X_test_vect: 
        if v.size: 
            X_test_vect_avg.append(v.mean(axis=0)) 
        else: 
            X_test_vect_avg.append(np.zeros(100, dtype =float))
            
    return X_train_vect_avg,X_test_vect_avg


# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (x_word2vec['N_text_clean'], x_word2vec['Score'] , test_size=0.2)


# In[21]:


X_train


# In[22]:


num_vector = 100
w2v_model = gensim.models.Word2Vec(X_train,window=5, min_count=2,vector_size = num_vector)
words = set(w2v_model.wv.index_to_key) 
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) 
                     for ls in X_train]) 
X_test_vect = np.array( [np.array([w2v_model.wv[i] for i in ls if i in words]) 
                     for ls in X_test])

# 通過平均句子中包含的單詞的單詞向量來計算句子向量
X_train_vect_avg = [] 
for v in X_train_vect: 
    if v.size: 
        X_train_vect_avg.append(v.mean(axis=0)) 
    else: 
        X_train_vect_avg.append(np.zeros(100, dtype=float)) 

X_test_vect_avg = [] 
for v in X_test_vect: 
    if v.size: 
        X_test_vect_avg.append(v.mean(axis=0)) 
    else: 
        X_test_vect_avg.append(np.zeros(100, dtype =float))


# ### 建構訓練模型

# In[26]:


from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

token = Tokenizer(num_words=5000)  ## 建立字典，總長5000
token.fit_on_texts(X_train) 
token.word_index

#透過texts_to_sequences可以將訓練和測試集資料中的影評文字轉換為數字list
x_train_seq = token.texts_to_sequences(X_train)
x_test_seq = token.texts_to_sequences(X_test)

print("train data size:",len(x_train_seq),"  test data size:",len(x_test_seq))


# In[76]:


x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)


# x_train = sequence.pad_sequences(X_train_vect_avg, maxlen=380)
# x_test = sequence.pad_sequences(X_test_vect_avg, maxlen=380)

print("train data size:",x_train.shape,"  test data size:",x_test.shape)


# In[77]:


x_test


# ### CNN

# In[30]:


from keras import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D  #匯入layers模組
from keras.layers import ZeroPadding2D,Activation  #匯入layers模組
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer


# In[78]:


# define model
inputshape = (len(x_train),1)

modelCNN = Sequential()
modelCNN.add(Embedding(output_dim=32,
                        input_dim=5000,
                        input_length=x_train.shape[1]))
modelCNN.add(Conv1D(filters=32, kernel_size=8, activation='relu',input_shape =(x_train.shape[0],1) ))
modelCNN.add(MaxPooling1D(pool_size=2))
modelCNN.add(Flatten())
modelCNN.add(Dense(10, activation='relu'))
modelCNN.add(Dense(1, activation='sigmoid'))

modelCNN.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

modelCNN.build(x_train.shape)

modelCNN.summary()


# In[79]:


train_history = modelCNN.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)


# In[80]:


scores = modelCNN.evaluate(x_test, y_test,verbose=1)
scores[1]


# ### LSTM

# In[82]:


modelLSTM = Sequential() #建立模型

modelLSTM.add(Embedding(output_dim=32,#輸出的維度是32，希望將數字list轉換為32維度的向量
                        input_dim=5000,#輸入的維度是5000，也就是我們之前建立的字典是5000字
                        input_length=x_train.shape[1])) #數字list截長補短後都是380個數字

modelLSTM.add(Dropout(0.7)) #隨機


# In[85]:


# modelLSTM.add(Flatten(input_shape = (28,28)))
modelLSTM.add(LSTM(32,input_shape = x_train.shape,return_sequences = True)) 
modelLSTM.add(Dense(units=256,activation='relu')) 
modelLSTM.add(Dropout(0.7))
modelLSTM.add(Dense(units=1,activation='sigmoid'))

modelLSTM.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

modelLSTM.build((x_train.shape))
modelLSTM.summary()


# In[86]:


train_history = modelLSTM.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)


# In[87]:


scores = modelLSTM.evaluate(x_test, y_test,verbose=1)
scores[1]


# In[ ]:




