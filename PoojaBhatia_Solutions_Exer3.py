#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
import re

# parameters
max_fatures = 500
embed_dim = 128
lstm_out = 196
dropout = 0.1
dropout_1d = 0.4
recurrent_dropout = 0.1
random_state = 1324
validation_size = 1000
batch_size = 16
epochs=2
verbose= 2

# Preprocess and Read Data 
df = pd.read_csv('dataset_sentiment.csv')
df = df[['text','sentiment']]
print(df[0:10])

df = df[df.sentiment != "Neutral"] 
df['text'] = df['text'].apply(lambda x: x.lower()) #
df['text'] = df['text'].apply(lambda x: x.replace('rt',' '))
df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
print(df[0:10])    
    
tok = Tokenizer(num_words=max_fatures, split=' ')
tok.fit_on_texts(df['text'].values)
X = tok.texts_to_sequences(df['text'].values)
X = pad_sequences(X)

# Model1:Using LSTM
def model_1():
    nn = Sequential()
    nn.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
    nn.add(SpatialDropout1D(dropout_1d))
    nn.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=recurrent_dropout))
    nn.add(Dense(2, activation='softmax'))
    nn.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(nn.summary())
    return nn

#Model2:Using ConvNet
def model_2():
    nn = Sequential()
    nn.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
    nn.add(Convolution1D(filters=100,kernel_size=3, padding="valid", activation="relu", strides=1))
    nn.add(MaxPooling1D(pool_size=2))
    nn.add(Flatten())
    nn.add(Dense(2, activation='softmax'))
    nn.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(nn.summary())
    return nn

Y = pd.get_dummies(df['sentiment']).values

#Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = random_state)


X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

#Evaluation Function 
def evaluation(nn):
    
    score, accuracy = nn.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (accuracy))

    pos_cnt, neg_cnt, pos_ok, neg_ok = 0, 0, 0, 0
    for x in range(len(X_validate)):
        result = nn.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0: neg_ok += 1
            else: pos_ok += 1
        if np.argmax(Y_validate[x]) == 0: neg_cnt += 1
        else: pos_cnt += 1

    print("pos_acc", pos_ok/pos_cnt*100, "%")
    print("neg_acc", neg_ok/neg_cnt*100, "%")

    X2 = ['what are u going to say about that? the truth, wassock?!']
    X2 = tok.texts_to_sequences(X2)
    X2 = pad_sequences(X2, maxlen=26, dtype='int32', value=0)
    print(X2)
    print(nn.predict(X2, batch_size=1, verbose = 2)[0])



####Results####

nn_1=model_1()
nn_1.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, verbose=verbose)
evaluation(nn_1)


nn_2=model_2()
nn_2.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, verbose=verbose)
evaluation(nn_2)



# In[8]:


#Generating MetaData

from rdflib import Namespace, Graph, Literal
from rdflib.namespace import FOAF, OWL, XSD, RDFS, DCTERMS, DOAP, DC, RDF


prov = Namespace('http://www.w3.org/ns/prov#')
dcat = Namespace('http://www.w3.org/ns/dcat#')
mexalgo = Namespace('http://mex.aksw.org/mex-algo#')
mexperf = Namespace('http://mex.aksw.org/mex-perf#')
mexcore = Namespace('http://mex.aksw.org/mex-core#')
this = Namespace('http://mex.aksw.org/examples/')

g = Graph()
# Create Binding
g.bind('dct',DCTERMS)
g.bind('owl',OWL)
g.bind('foaf',FOAF)
g.bind('xsd', XSD)
g.bind('rdfs', RDFS)
g.bind('doap', DOAP)
g.bind('dc', DC)
g.bind('prov', prov)
g.bind('dcat', dcat)
g.bind('mexalgo',mexalgo)
g.bind('mexperf',mexperf)
g.bind('mexcore',mexcore)
g.bind('this',this)


g.add((this.pooja_task3,mexcore.Experiment, prov.Entity))
g.add((this.pooja_task3,mexcore.ApplicationContext, prov.Entity))
g.add((this.pooja_task3,DCTERMS.date, Literal('2018-07-22',datatype=XSD.date)))
g.add((this.pooja_task3,FOAF.givenName, Literal('Pooja Bhatia')))
g.add((this.pooja_task3,FOAF.mbox, Literal('pooja12.3.92@gmail.com')))


#Configuration of Model 1
g.add((this.configuration1,RDF.type,mexcore.ExperimentConfiguration))
g.add((this.configuration1,prov.used, this.model1))
g.add((this.configuration1,prov.wasStartedBy,this.pooja_task3))

#Configuration of Model 2
g.add((this.configuration2,RDF.type,mexcore.ExperimentConfiguration))
g.add((this.configuration2,prov.used, this.model2))
g.add((this.configuration2,prov.wasStartedBy,this.pooja_task3))




g.add((this.hyerparameter_model1,mexalgo.HyperParameterCollection,prov.Entity))
g.add((this.hyerparameter1,RDFS.label,Literal('HyperParameterCollection')))
g.add((this.hyerparameter_model1,prov.hadMember,this.hyerparameter1))

g.add((this.hyerparameter_model2,mexalgo.HyperParameterCollection,prov.Entity))
g.add((this.hyerparameter2,RDFS.label,Literal('HyperParameterCollection')))
g.add((this.hyerparameter_model2,prov.hadMember,this.hyerparameter2))


g.add((this.hyerparameter1,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter1,RDFS.label, Literal('LSTM')))
g.add((this.hyerparameter1,DCTERMS.identifier, Literal('LSTM')))
g.add((this.hyerparameter1,prov.value, Literal('196',datatype=XSD.float)))


g.add((this.hyerparameter2,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter2,RDFS.label, Literal('ConvNet')))
g.add((this.hyerparameter2,DCTERMS.identifier, Literal('ConvNet')))
g.add((this.hyerparameter2,prov.value, Literal('100',datatype=XSD.float)))


g.add((this.execution1,mexcore.ExecutionOverall,prov.Entity))
g.add((this.execution1,prov.generated,this.performance_measures1))
g.add((this.execution1,prov.used,this.test))
g.add((this.execution1,prov.used,this.hyerparameter_model1))
g.add((this.execution1,prov.used,this.model1))

g.add((this.performance_measures1,mexcore.PerformanceMeasure,prov.Entity))
g.add((this.performance_measures1,mexperf.score,Literal('0.38',datatype=XSD.float)))
g.add((this.performance_measures1,mexperf.accuracy,Literal('0.84',datatype=XSD.float)))
g.add((this.performance_measures1,prov.wasGeneratedBy,this.execution1))


g.add((this.execution2,mexcore.ExecutionOverall,prov.Entity))
g.add((this.execution2,prov.generated,this.performance_measures2))
g.add((this.execution2,prov.used,this.test))
g.add((this.execution2,prov.used,this.model2))

g.add((this.performance_measures2,mexcore.PerformanceMeasure,prov.Entity))
g.add((this.performance_measures2,mexperf.score,Literal('0.38',datatype=XSD.float)))
g.add((this.performance_measures2,mexperf.accuracy,Literal('0.85',datatype=XSD.float)))
g.add((this.performance_measures2,prov.wasGeneratedBy,this.execution2))


g.add((this.model1,mexalgo.Algorithm,prov.Entity))
g.add((this.model1,RDFS.label,Literal('LSTM')))
g.add((this.model1,mexalgo.hasHyperParameter,this.hyerparameter1))

g.add((this.model2,mexalgo.Algorithm,prov.Entity))
g.add((this.model2,RDFS.label,Literal('ConvNet')))
g.add((this.model2,mexalgo.hasHyperParameter,this.hyerparameter2))


with open('pooja_Exer3_metadata.ttl','wb') as f:
    f.write(g.serialize(format='turtle'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




