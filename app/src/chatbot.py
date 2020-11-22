# -*- coding: utf-8 -*-
"""Copy of python_chatbot_example_with_tensorflow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jl4iTFi_-ac9ZxC4uX_SDS8S13ibbN13
"""

##########################################################################################
# "How to make a simple contextualized chatbot with tensorflow, keras, nltk and sklearn"
# 
#  by Andreas S. Rath <andreas.rath@ondewo.com> 
#  Github name: teddius
#  Github source code: http://bit.ly/tfcb17ondewo
#
#  Inspired by chatbotsmagazine article which was based on "tflearn" and is available at 
#  https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
##########################################################################################

######################################################################
# basic things we need for python processing and google colab
######################################################################

import random
import base64
import requests
import numpy as np
import json
#from google.colab import files

######################################################################
# Things we need for NLP
######################################################################
import nltk
nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() # english stemmer

######################################################################
# things we need for Tensorflow
######################################################################
import tensorflow as tf
print(tf.__version__)

from keras import metrics, optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, Flatten, Conv1D, Embedding, MaxPooling1D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# data_url = "https://raw.githubusercontent.com/rupel190/voiceinterfaceprogramming/main/intents.json"
# intent_json = req.text
# print(intent_json)
# req = requests.get(data_url)

data_url = "./intents.json"
intent_json = open(data_url, 'r').read()
intents = json.loads(intent_json)
print(intents)


######################################################################
# Import our chat-bot intents file
######################################################################

intents = json.loads(intent_json)
print(intents)

######################################################################
# Let's start to build our training data
######################################################################
words = []
classes = []
documents = []
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
      
        # add to our words list
        words.extend(w)
        
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print('------------------------------------------------------')
print('-------------  Summary -------------------------------')
print('------------------------------------------------------')
print('')
print(len(classes), "classes\n", classes)
print('')
print('------------------------------------------------------')
print('')
print(len(words), "words\n", words)
print('')
print('------------------------------------------------------')
print('')
print(len(documents), "documents\n", documents)
print('')
print('------------------------------------------------------')
print('')
print(len(words), "unique stemmed words\n", words)
print('')

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    
# print('training: ' + str(training))

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
X = list(training[:, 0])
y = list(training[:, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=32, shuffle=True)

print('------------------------------------------------------')
print('-------------  Summary -------------------------------')
print('------------------------------------------------------')
print('')
print('Total elements in X: ' + str(len(X)) + ' consisting of') 
print('elements in X_train: ' + str(len(X_train)) + ' and X_test: ' + str(len(X_test)))
print('')
print('Total elements in y (labels): ' + str(len(y)) + ' consisting of') 
print('elements in y_train: ' + str(len(y_train)) + ' and y_test: ' + str(len(y_test)))

print('------------------------------------------------------')
print('----- Let us look a specific training example --------')
print('------------------------------------------------------')
print('X_train[0] (bag of word references):', X_train[0])
print('------------------------------------------------------')
print('y_train[0] (class label):', y_train[0])
print('------------------------------------------------------')
print('All class labels:', classes)
print('------------------------------------------------------')
print('Our training example class label at index classes[' + str(y_train[0].index(1))+ ']=',
      classes[y_train[0].index(1)])  # TODO show clas

####################################################################################
# Build a very simple neural network
####################################################################################
model = Sequential()
model.add(Dense(100, activation="relu",input_dim=(np.array(X_train).shape[1])))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(classes), activation='softmax'))

# metrics
adam = optimizers.Adam(lr=0.1, decay=0.005)
model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
print(model.summary())

####################################################################################
# OPTIONAL for playing around you could add the following layers (watch out to 
# transform to correct shape)
# 
# model.add(Embedding(len(words), embedding_vector_length))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
####################################################################################

# Callbacks for the evaluation of the model
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
tensorboard_callback = TensorBoard(log_dir='./logs/')
checkpoint = ModelCheckpoint('./weights-improvement-{epoch:02d}-{loss:.4f}.hdf5',
                             monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint, tensorboard_callback, early_stop]
callbacks_list = [tensorboard_callback, early_stop]

nr_of_epoches=100
batch_size=32
history = model.fit(X_train,
                    y_train,
                    epochs=nr_of_epoches,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks_list)

#############################################################
# Let's have a look at a single test example from X_test 
#############################################################
print('')
print('X_test[0] in total looks like:\n\n', X_test[0])
print('')
print('------------------------------------------------------')
print('')
print('X_test[0] has the class index stored at y_test[0] with the label: classes[' + str(y_test[0].index(1))+ ']=',
      classes[y_test[0].index(1)]) 
print('')
print('------------------------------------------------------')
print('')
prediction = model.predict(np.array([X_test[0]]))
print('Total "raw" prediction for all classes looks like:\n\n', prediction)

ERROR_THRESHOLD = 0.00000000001
# generate probabilities from the model
results = [[i, r] for i, r in enumerate(prediction[0]) if r > ERROR_THRESHOLD]
print('Our prediction translated to classes and probabilities:\n')
for r in results:
    print(classes[r[0]], round(r[1], 8))

###################################################################################
# Let's define two needy functions to do the natural language preprocessing for us
# and build the bag of words (bow) for us from a sentence of words
###################################################################################
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

print('These are all the words for our classification:\n\n', words)
print('')
print('Our sentence is represented by the following bag of words (bow):\n')
p = bow("Can you please tell me if you are open today?", words, show_details=True)

print('Kindly reminder: our classes we want to predict are:\n\n', classes)

ERROR_THRESHOLD=0.0001
prediction = model.predict(np.array([p]))
# generate probabilities from the model
results = [[i, r] for i, r in enumerate(prediction[0]) if r > ERROR_THRESHOLD]

print('\nOur prediction is:\n')
for r in results:
    print('=> ', classes[r[0]], round(r[1],4))

########################################################################
# Let's create a needy data structure to 
# (1) hold and track the user context
# (2) classifies our sentence to a class
# (3) generates a contextualized response for a specific user 
#        based on 3 elements
#       (a) class with highest prediction propability
#       (b) a specific user id
#       (c) context set
########################################################################

# (1) hold and track the user context
context = {}


# (2) classifies our sentence to a class
def classify(sentence):
    # generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

  

# (3) generates a contextualized response for a specific user
def response(sentence, user_id='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                  
                  # check if this intent is contextual and applies to this user's conversation
                  if not 'context_filter' in i or (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                      if show_details: print ('tag:', i['tag'])
                      # a random response from the intent
                      response = (random.choice(i['responses']))

                  # set context if available
                  if 'context_set' in i:
                      if show_details: print ('context:', i['context_set'])
                      context[user_id] = i['context_set']
                                    
                  return print(response)

            results.pop(0)

#response('Hey')
#classify('fruit')
response('Can you place something?')
# So how does this context thing work?
#context = {}
#response('Can you place something?', user_id='rupel', show_details=True)
#print(context)
#response('Move an apple', user_id='rupel', show_details=True)
#response('a table', user_id='rupel',show_details=True)
#response('behind')
#response('bye')