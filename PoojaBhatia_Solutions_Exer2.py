#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Mon May 07 19:27:08 2018

@author: POOJA
"""
import nltk 
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk import DefaultTagger as df
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
from nltk import TrigramTagger as tg
import plotly.plotly as py
import plotly.graph_objs as go
from nltk.corpus import TaggedCorpusReader
import plotly.offline as offline
from plotly.offline import init_notebook_mode, iplot 

from IPython.display import display


#### Corpus X1#####
treebank_annotated_sent = nltk.corpus.treebank.tagged_sents()
sizeX1= int(len(treebank_annotated_sent)* 0.8)
train_sents_treebank = treebank_annotated_sent[:sizeX1]
test_sents_treebank = treebank_annotated_sent[sizeX1:]

####Corpus X2####
brown_annotated_sent = nltk.corpus.brown.tagged_sents()
sizeX2 = int(len(brown_annotated_sent) * 0.8)
train_sents_brown = brown_annotated_sent[:sizeX2]
test_sents_brown = brown_annotated_sent[sizeX2:]

################################ MODEL 1####################################################
#####Training#######
def features(sentence, index):
    return {
        'word': sentence[index],
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_entire_word_capitalized':sentence[index].upper() == sentence[index],
        'prefix-1': sentence[index][0],
        'suffix-1': sentence[index][-1],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'length_word': len(sentence[index]),
         'is_numeric': sentence[index].isdigit(),
         'is_alphabetic': sentence[index].isalpha(),
         'is_alphanumeric':sentence[index].isalnum(),
    }
def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    X, y = [], []
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y
 
X_treebank, y_treebank = transform_to_dataset(train_sents_treebank)    
X_brown, y_brown = transform_to_dataset(train_sents_brown)
#########Implementing a classifier#############################
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

size=10000

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])
####for treebank###
clf.fit(X_treebank[:size],y_treebank[:size])
 
print('training OK for X1')
 
X_treebank_test, y_treebank_test = transform_to_dataset(test_sents_treebank)

####for Brown###
clf.fit(X_brown[:size],y_brown[:size])
 
print('training OK for X2')
 
X_brown_test, y_brown_test = transform_to_dataset(test_sents_brown)
print()

####################################### MODEL 2###########################################
nltk.download('maxent_treebank_pos_tagger')
MAXEXT_POS_TAGGER =nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
#################Model3.x = rule-based classifiers (x = 1 to 5)##########################
patterns = [(r'.*ing$', 'VBG'), (r'.*ed$', 'VBD'), (r'.*es$', 'VBZ'), (r'.*ould$', 'MD'), (r'.*\'s$', 'NN$'),               
             (r'.*s$', 'NNS'), (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), (r'.*', 'NN')]

###Training OF model3.x in X1######
def_model_1 = nltk.DefaultTagger('NN')
uni_model_1 = nltk.UnigramTagger(train_sents_treebank)
bi_model_1 = nltk.BigramTagger(train_sents_treebank)
tri_model_1 = nltk.TrigramTagger(train_sents_treebank)
regexp_model_1 = nltk.RegexpTagger(patterns)
###Training OF model3.x in X2######
def_model_2 = nltk.DefaultTagger('NN')
uni_model_2 = nltk.UnigramTagger(train_sents_brown)
bi_model_2 = nltk.BigramTagger(train_sents_brown)
tri_model_2 = nltk.TrigramTagger(train_sents_brown)
regexp_model_2 = nltk.RegexpTagger(patterns)
#########TASK 1####################################################

########performance 1.1 = model1 in X1#############################
print("performance 1.1 = model1 in X1")
performance_1_1 =clf.score(X_treebank_test, y_treebank_test)
print("Accuracy:", performance_1_1)
print()
########performance 1.2 = model2 in X1#############################
print("performance 1.2 = model2 in X1")
performance_1_2= MAXEXT_POS_TAGGER.evaluate(treebank_annotated_sent)
print("Accuracy:",performance_1_2)
print()
########performance 1.3.x = model3.x in X1#########################
# performance of Default Tagger
print("performance 1.3.1 = model3.1 in X1")
performance_1_3_1= def_model_1.evaluate(test_sents_treebank)
print("Accuracy:",performance_1_3_1)
print()
# performance of Unigram Tagger
print("performance 1.3.2 = model3.2 in X1")
performance_1_3_2=uni_model_1.evaluate(test_sents_treebank)
print("Accuracy:",performance_1_3_2)
print()
# performance of Bigram Tagger
print("performance 1.3.3 = model3.3 in X1")
performance_1_3_3=bi_model_1.evaluate(test_sents_treebank)
print("Accuracy:",performance_1_3_3)
print()
# performance of Trigram Tagger
print("performance 1.3.4 = model3.4 in X1")
performance_1_3_4=tri_model_1.evaluate(test_sents_treebank)
print("Accuracy:",performance_1_3_4)
print()
# performance of Regex Tagger
print("performance 1.3.5 = model3.5 in X1")
performance_1_3_5=regexp_model_1.evaluate(test_sents_treebank)
print("Accuracy:",performance_1_3_5)
print()

########performance 1.4 = model1 in X2#######################
print("performance 1.4 = model1 in X2")
performance_1_4 =clf.score(X_brown_test, y_brown_test)
print("Accuracy:", performance_1_4)
print()
#######performance 1.5 = model2 in X2#######################
print("performance 1.5 = model2 in X2 ")
performance_1_5= MAXEXT_POS_TAGGER.evaluate(brown_annotated_sent)
print("Accuracy:",performance_1_5)
print()
########performance 1.6.x = model3.x in X2########
# performance of Default Tagger
print("performance 1.6.1 = model3.1 in X2")
performance_1_6_1= def_model_2.evaluate(test_sents_brown)
print("Accuracy:",performance_1_6_1)
print("")
# performance of Unigram Tagger
print("performance 1.6.2 = model3.2 in X2")
performance_1_6_2=uni_model_2.evaluate(test_sents_brown)
print("Accuracy:",performance_1_6_2)
print("")
# performance of Bigram Tagger
print("performance 1.6.3 = model3.3 in X2")
performance_1_6_3=bi_model_2.evaluate(test_sents_brown)
print("Accuracy:",performance_1_6_3)
print("")
# performance of Trigram Tagger
print("performance 1.6.4 = model3.4 in X2")
performance_1_6_4=tri_model_2.evaluate(test_sents_brown)
print("Accuracy:",performance_1_6_4)
print("")
# performance of Regex Tagger
print("performance 1.6.5 = model3.5 in X2")
performance_1_6_5=regexp_model_2.evaluate(test_sents_brown)
print("Accuracy:",performance_1_6_5)
print("")

######## Results of Task1 on BarChart###########
data = [go.Bar(
            x=['Task 1.1', 'Task 1.2', 'Task 1.3.1','Task 1.3.2', 'Task 1.3.3','Task 1.3.4','Task 1.3.5',
              'Task 1.4','Task 1.5', 'Task 1.6.1','Task 1.6.2', 'Task 1.6.3','Task 1.6.4','Task 1.6.5'],
            y=[performance_1_1, performance_1_2, performance_1_3_1,performance_1_3_2, performance_1_3_3, performance_1_3_4,
               performance_1_3_5, performance_1_4, performance_1_5,performance_1_6_1, performance_1_6_2, performance_1_6_3,
               performance_1_6_4, performance_1_6_5]
    )]
layout = go.Layout(
    title='Results of Task1 on BarChart',
    xaxis=dict(
         title='Task Number',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Accuracy',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        )
    ))
figure=go.Figure(data=data, layout=layout)
offline.plot(figure, image='png', filename='Task1.html')

####Corpus X3#########
corp = nltk.corpus.ConllCorpusReader('.', 'germanfile',
                                     ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                     encoding='utf-8')

tagged_sents = corp.tagged_sents()

sizeX3 = int(len(tagged_sents) * 0.8)
train_tagged_sents = tagged_sents[:sizeX3]
test_tagged_sents = tagged_sents[sizeX3:]

####Model 4############
X_tagged, y_tagged = transform_to_dataset(train_tagged_sents)
size=10000

clf.fit(X_tagged[:size],y_tagged[:size])
 
print('training OK for X3')
 
X_tagged_test, y_tagged_test = transform_to_dataset(test_tagged_sents)


########performance 2.1 = model4 in X3#############################
print("performance 2.1 = model4 in X3")
performance_2_1 =clf.score(X_tagged_test, y_tagged_test)
print("Accuracy:", performance_2_1)
print()

####Model 5############
#import os
#os.environ['TREETAGGER_HOME'] = '/Users/POOJA/Documents/TreeTagger/cmd'

from treetagger import TreeTagger
tt = TreeTagger(language='german')
#result_train=tt.tag(X_tagged_test)

########performance 2.2 = model5 in X3#############################

print("performance 2.2 = model5 in X3")
#performance_2_2 = np.mean([x[1] == y for x, y in zip(res_train, y_tagged_test)])
###STORING Accuracy has 0 because TreeTagger is giving AttributeError: 'TreeTagger' object has no attribute '_treetagger_bin' 
#####error in the testtagger.py file--Unable to create environment in Windows 10 for the same########

performance_2_2 = 0.0
print("Accuracy:", performance_2_2)
print()

######## Results of Task1 on BarChart###########
data = [go.Bar(
            x=['Task 2.1', 'Task 2.2'],
            y=[performance_2_1, performance_2_2]
    )]
layout = go.Layout(
    title='Results of Task2 on BarChart',
    xaxis=dict(
         title='Task Number',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Accuracy',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        )
    ))
figure=go.Figure(data=data, layout=layout)

offline.plot(figure, image='png', filename='Task2.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




