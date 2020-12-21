


######################################################################
# basic things we need for python processing and google colab
######################################################################

import random
import base64
import requests
import numpy as np
import json
from google.colab import files

######################################################################
# Things we need for NLP
######################################################################
import nltk
nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() # english stemmer

# Import Spacy library
import spacy
from spacy.lookups import Lookups

# load the English web corpus small which we just downloaded 
nlp = spacy.load("en_core_web_sm")

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


# Speelinng Correction
from symspellpy.symspellpy import SymSpell, Verbosity
from tqdm.notebook import tqdm

import re, string, json
from itertools import islice
import itertools
import pkg_resources

    
    
    

######################################################################
# Post-process intents to generate patterns
# (Learn [learn] is a mnemonic creates combinational duplicates which aren't removed but "Learn [learn] was removed instead from the intents.")
######################################################################

def patternCombinator(intents):
  newIntents = intents
  replacements = extractReplacements(newIntents)
  print('\n\nAvailable replacements: ', replacements, '\n')
  for intent in newIntents['intents']:
    #print("Generating for intent: ", newIntents['patterns'])
    generatedPatterns = [ ]
    for pattern in intent['patterns']:
      listOfPlaceholderValues = replacementLists(intents, pattern, replacements)
      #print('Replacements for sentence: ', listOfPlaceholderValues)
      generatedValues = combineLists(listOfPlaceholderValues)
      #print('Generated Values: ', generatedValues, '\n')
      sentences = generatePatterns(pattern, generatedValues)
      #print('Generated patterns: ', sentences)
      generatedPatterns.append(sentences)
    generatedPatterns = list(itertools.chain(*generatedPatterns))
    intent['patterns'] = generatedPatterns
    print("Done for intent: ", intent['patterns'])
  return newIntents

#private
# find types (e.g. fruits) and values (e.g. apple)
def extractReplacements(intents):
  replacements = { }
  for replacementType in intents['replacements']:
    replacementValues = intents['replacements'][replacementType]
    #print('Replacement Type: ', replacementType, ' has values: ', replacementValues)
    replacements.setdefault(replacementType, replacementValues)
  return replacements

#private
def replacementLists(intents, pattern, replacements):
  import re
  expression = r"\[\S*?\]" # match valid only inside brackets
  listOfPlaceholderValues = [ ]
  # for each placeholder in the sentence find the type
  for placeholder in re.finditer(expression, pattern):
    placeholderValues = [ ]
    # save list for the values of each type
    replacementType = placeholder.group().replace('[', '').replace(']', '')
    # special case for empty brackets
    if replacementType is '': 
      # use the replacementtype structure here
      placeholderValues = intents['replacementTypes']
    else:
      placeholderValues = replacements[replacementType]
    #print('Replacement TYPE: ', replacementType)
    #print('Replacement VALUES: ', placeholderValues)
    listOfPlaceholderValues.append(placeholderValues)
  return listOfPlaceholderValues

#private
def combineLists(listOfPlaceholderValues):
  import itertools
  combinations = list(itertools.product(*listOfPlaceholderValues)) # those (e.g.) 3 lists
  #print('Combinations: ', *combinations, sep = "\n")
  return combinations

#private
def generatePatterns(pattern, values):
  import re
  expression = r"\[\S*?\]" # match valid only inside brackets
  print('Generating for pattern: ', pattern)
  print('Generating for values: ', values)
  sentences = []
  for replacement_row in values:
    #print('Replacement row: ', replacement_row)
    sentence = pattern
    for index, match in enumerate(re.finditer(expression, pattern)):
      replacement = '[' + replacement_row[index] + ', ' + match.group().replace('[','').replace(']','') + ']'
      #print('Match: ', match.group(), ' with index: ', index, 'is going to be replaced with', replacement)
      sentence = sentence.replace(match.group(), replacement)
    #print('Resulting sentence: ', sentence)
    sentences.append(sentence)
  return sentences

#TRAIN_SINGLE = [('Put the plum onto the bed', {'entities': [(8, 12, 'fruit'), (22, 25, 'place')]})]

# For testing
#testpattern = '[direction] of the [place] is a good spot for a [fruit]'
#listOfPlaceholderValues = replacementLists(testpattern, replacementDicta(intents))
#combineLists(listOfPlaceholderValues)
#content = [('apple', 'up', 'desk'), ('apple', 'up', 'table'), ('apple', 'up', 'box'), ('apple', 'up', 'shelf'), ('apple', 'up', 'ground'), ('apple', 'down', 'desk'), ('apple', 'down', 'table'), ('apple', 'down', 'box'), ('apple', 'down', 'shelf'), ('apple', 'down', 'ground'), ('apple', 'left', 'desk'), ('apple', 'left', 'table'), ('apple', 'left', 'box'), ('apple', 'left', 'shelf'), ('apple', 'left', 'ground'), ('apple', 'right', 'desk'), ('apple', 'right', 'table'), ('apple', 'right', 'box'), ('apple', 'right', 'shelf'), ('apple', 'right', 'ground'), ('apple', 'on the left', 'desk'), ('apple', 'on the left', 'table'), ('apple', 'on the left', 'box'), ('apple', 'on the left', 'shelf'), ('apple', 'on the left', 'ground'), ('apple', 'on the right', 'desk'), ('apple', 'on the right', 'table'), ('apple', 'on the right', 'box'), ('apple', 'on the right', 'shelf'), ('apple', 'on the right', 'ground'), ('apple', 'top', 'desk'), ('apple', 'top', 'table'), ('apple', 'top', 'box'), ('apple', 'top', 'shelf'), ('apple', 'top', 'ground'), ('apple', 'on top', 'desk'), ('apple', 'on top', 'table'), ('apple', 'on top', 'box'), ('apple', 'on top', 'shelf'), ('apple', 'on top', 'ground'), ('apple', 'under', 'desk'), ('apple', 'under', 'table'), ('apple', 'under', 'box'), ('apple', 'under', 'shelf'), ('apple', 'under', 'ground'), ('apple', 'below', 'desk'), ('apple', 'below', 'table'), ('apple', 'below', 'box'), ('apple', 'below', 'shelf'), ('apple', 'below', 'ground'), ('apple', 'on the bottom', 'desk'), ('apple', 'on the bottom', 'table'), ('apple', 'on the bottom', 'box'), ('apple', 'on the bottom', 'shelf'), ('apple', 'on the bottom', 'ground'), ('banana', 'up', 'desk'), ('banana', 'up', 'table'), ('banana', 'up', 'box'), ('banana', 'up', 'shelf'), ('banana', 'up', 'ground'), ('banana', 'down', 'desk'), ('banana', 'down', 'table'), ('banana', 'down', 'box'), ('banana', 'down', 'shelf'), ('banana', 'down', 'ground'), ('banana', 'left', 'desk'), ('banana', 'left', 'table'), ('banana', 'left', 'box'), ('banana', 'left', 'shelf'), ('banana', 'left', 'ground'), ('banana', 'right', 'desk'), ('banana', 'right', 'table'), ('banana', 'right', 'box'), ('banana', 'right', 'shelf'), ('banana', 'right', 'ground'), ('banana', 'on the left', 'desk'), ('banana', 'on the left', 'table'), ('banana', 'on the left', 'box'), ('banana', 'on the left', 'shelf'), ('banana', 'on the left', 'ground'), ('banana', 'on the right', 'desk'), ('banana', 'on the right', 'table'), ('banana', 'on the right', 'box'), ('banana', 'on the right', 'shelf'), ('banana', 'on the right', 'ground'), ('banana', 'top', 'desk'), ('banana', 'top', 'table'), ('banana', 'top', 'box'), ('banana', 'top', 'shelf'), ('banana', 'top', 'ground'), ('banana', 'on top', 'desk'), ('banana', 'on top', 'table'), ('banana', 'on top', 'box'), ('banana', 'on top', 'shelf'), ('banana', 'on top', 'ground'), ('banana', 'under', 'desk'), ('banana', 'under', 'table'), ('banana', 'under', 'box'), ('banana', 'under', 'shelf'), ('banana', 'under', 'ground'), ('banana', 'below', 'desk'), ('banana', 'below', 'table'), ('banana', 'below', 'box'), ('banana', 'below', 'shelf'), ('banana', 'below', 'ground'), ('banana', 'on the bottom', 'desk'), ('banana', 'on the bottom', 'table'), ('banana', 'on the bottom', 'box'), ('banana', 'on the bottom', 'shelf'), ('banana', 'on the bottom', 'ground'), ('kiwi', 'up', 'desk'), ('kiwi', 'up', 'table'), ('kiwi', 'up', 'box'), ('kiwi', 'up', 'shelf'), ('kiwi', 'up', 'ground'), ('kiwi', 'down', 'desk'), ('kiwi', 'down', 'table'), ('kiwi', 'down', 'box'), ('kiwi', 'down', 'shelf'), ('kiwi', 'down', 'ground'), ('kiwi', 'left', 'desk'), ('kiwi', 'left', 'table'), ('kiwi', 'left', 'box'), ('kiwi', 'left', 'shelf'), ('kiwi', 'left', 'ground'), ('kiwi', 'right', 'desk'), ('kiwi', 'right', 'table'), ('kiwi', 'right', 'box'), ('kiwi', 'right', 'shelf'), ('kiwi', 'right', 'ground'), ('kiwi', 'on the left', 'desk'), ('kiwi', 'on the left', 'table'), ('kiwi', 'on the left', 'box'), ('kiwi', 'on the left', 'shelf'), ('kiwi', 'on the left', 'ground'), ('kiwi', 'on the right', 'desk'), ('kiwi', 'on the right', 'table'), ('kiwi', 'on the right', 'box'), ('kiwi', 'on the right', 'shelf'), ('kiwi', 'on the right', 'ground'), ('kiwi', 'top', 'desk'), ('kiwi', 'top', 'table'), ('kiwi', 'top', 'box'), ('kiwi', 'top', 'shelf'), ('kiwi', 'top', 'ground'), ('kiwi', 'on top', 'desk'), ('kiwi', 'on top', 'table'), ('kiwi', 'on top', 'box'), ('kiwi', 'on top', 'shelf'), ('kiwi', 'on top', 'ground'), ('kiwi', 'under', 'desk'), ('kiwi', 'under', 'table'), ('kiwi', 'under', 'box'), ('kiwi', 'under', 'shelf'), ('kiwi', 'under', 'ground'), ('kiwi', 'below', 'desk'), ('kiwi', 'below', 'table'), ('kiwi', 'below', 'box'), ('kiwi', 'below', 'shelf'), ('kiwi', 'below', 'ground'), ('kiwi', 'on the bottom', 'desk'), ('kiwi', 'on the bottom', 'table'), ('kiwi', 'on the bottom', 'box'), ('kiwi', 'on the bottom', 'shelf'), ('kiwi', 'on the bottom', 'ground'), ('pear', 'up', 'desk'), ('pear', 'up', 'table'), ('pear', 'up', 'box'), ('pear', 'up', 'shelf'), ('pear', 'up', 'ground'), ('pear', 'down', 'desk'), ('pear', 'down', 'table'), ('pear', 'down', 'box'), ('pear', 'down', 'shelf'), ('pear', 'down', 'ground'), ('pear', 'left', 'desk'), ('pear', 'left', 'table'), ('pear', 'left', 'box'), ('pear', 'left', 'shelf'), ('pear', 'left', 'ground'), ('pear', 'right', 'desk'), ('pear', 'right', 'table'), ('pear', 'right', 'box'), ('pear', 'right', 'shelf'), ('pear', 'right', 'ground'), ('pear', 'on the left', 'desk'), ('pear', 'on the left', 'table'), ('pear', 'on the left', 'box'), ('pear', 'on the left', 'shelf'), ('pear', 'on the left', 'ground'), ('pear', 'on the right', 'desk'), ('pear', 'on the right', 'table'), ('pear', 'on the right', 'box'), ('pear', 'on the right', 'shelf'), ('pear', 'on the right', 'ground'), ('pear', 'top', 'desk'), ('pear', 'top', 'table'), ('pear', 'top', 'box'), ('pear', 'top', 'shelf'), ('pear', 'top', 'ground'), ('pear', 'on top', 'desk'), ('pear', 'on top', 'table'), ('pear', 'on top', 'box'), ('pear', 'on top', 'shelf'), ('pear', 'on top', 'ground'), ('pear', 'under', 'desk'), ('pear', 'under', 'table'), ('pear', 'under', 'box'), ('pear', 'under', 'shelf'), ('pear', 'under', 'ground'), ('pear', 'below', 'desk'), ('pear', 'below', 'table'), ('pear', 'below', 'box'), ('pear', 'below', 'shelf'), ('pear', 'below', 'ground'), ('pear', 'on the bottom', 'desk'), ('pear', 'on the bottom', 'table'), ('pear', 'on the bottom', 'box'), ('pear', 'on the bottom', 'shelf'), ('pear', 'on the bottom', 'ground')] 
#print(generatePatterns('Put the [fruit] [direction] the [place]', content))

#print(*sentences, sep = "\n")

# reassign with generated intents

#re.sub(expression, 'values[0]', match.group())

####################################################################################
# Parse intents.json for Training Named Entity Recognition
# Example
# Input: "Take a [banana, fruit] and put it on the [table, place]"
# Intermediate State: "Take a banana and put it on the table"
# Output: "('Put the plum onto the bed', {'entities': [(8, 12, 'fruit'), (22, 25, 'place')]})"
####################################################################################
#pre-process training data with spacy nlp

#import spacy
#import re
#nlp = spacy.load("en_core_web_sm")
#entities = [ ]
#TRAIN_FULL = [ ]
#

def parseToGoldFormat(intents):
  TRAIN_FULL = [ ]

  for intent in intents['intents']:
    for pattern in intent['patterns']:
      training_line = goldFormat(pattern)
      TRAIN_FULL.append(training_line)
  return TRAIN_FULL

#private
def goldFormat(pattern):
  expression = r"\[(.*?)\]"
  #sentence = "Take a [banana, fruit] and put it on the [table, place]" # used for testing
  sentence = pattern
  doc = nlp(sentence)

  labels = [ ]  
  cleaned_sentence = removeEntityAnnotation(sentence) # Take a banana and put it on the table
  print('Pattern: ', pattern)
  for match in re.finditer(expression, doc.text):
    start, end = match.span()
    span = doc.char_span(start, end)      
    if span is not None:
      entityAnnotation = span.text.replace('[','').replace(']','').split(', ')
      entityValue = entityAnnotation[0] # banana
      entityType = entityAnnotation[1] # fruit
      if entityType == '':
        continue
      # find index for entityValue in the cleaned sentence
      startIndex = cleaned_sentence.find(entityValue)
      endIndex = startIndex+len(entityValue)
      print('entity value: ', entityValue)
      print('entity type: ', entityType)
      print('start index for ', entityValue, ': ', startIndex)
      print('end index for ', entityValue, ': ', endIndex)
      print('resulting substring: <', cleaned_sentence[startIndex:endIndex], '>', sep='')
      label  = [((startIndex, endIndex, entityType))]
      labels.extend(label)
      
  training_line = cleaned_sentence, {"entities": labels}

  print('trainling_line: ', training_line)
  return training_line;

#goldFormat("Take a [banana, fruit] and put it on the [table, place]")

####################################################################################
# Spelling Correction
####################################################################################

max_edit_distance_dictionary= 3 
prefix_length = 4

spellchecker = SymSpell(max_edit_distance_dictionary, prefix_length)

# Load word frequency dictionary 
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Print out first 5 elements to show dictionary is successfully loaded
print(list(islice(spellchecker.words.items(), 5)))

# Inspired by article https://towardsdatascience.com/text-normalization-7ecc8e084e31
# and examples taken from https://colab.research.google.com/drive/1U_C_4wAtlWQdaA84yVwHUCdkvQWEd7r9 

def _reduce_exaggerations(text):
  # Auxiliary function to help with exxagerated words.
  # Examples: woooooords -> words,  yaaaaaaaaaaaaaaay -> yay
  correction = str(text)
  return re.sub(r'([\w])\1+', r'\1', correction)

def is_numeric(text):
  for char in text:
    if not (char in "0123456789" or char in ",%.$"):
      return False
  return True

def spell_correction(sentence_list, max_edit_distance_dictionary= 3, prefix_length = 4):
  # Load word frequency dictionary 
  dictionary_path = pkg_resources.resource_filename(
  "symspellpy", "frequency_dictionary_en_82_765.txt")
  spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1)
  norm_sents = []
  print("Spell correcting")
  for sentence in tqdm(sentence_list):
      norm_sents.append(spell_correction_text(sentence,
                                              spellchecker,
                                              max_edit_distance_dictionary,
                                              prefix_length))
  return norm_sents

def spell_correction_text(text, 
                          spellchecker, 
                          max_edit_distance_dictionary= 3,
                          prefix_length = 4):
  """
  This function does very simple spell correction normalization using 
  pyspellchecker module. It works over a tokenized sentence and only the 
  token representations are changed.
  """
  if len(text) < 1:
      return ""
  #Spell checker config
  max_edit_distance_lookup = 2
  suggestion_verbosity = Verbosity.TOP # TOP, CLOSEST, ALL
  #End of Spell checker config
  token_list = text.split()
  for word_pos in range(len(token_list)):
      word = token_list[word_pos]
      if word is None:
          token_list[word_pos] = ""
          continue
      if not '\n' in word and word not in string.punctuation and not is_numeric(word) and not (word.lower() in spellchecker.words.keys()):
          suggestions = spellchecker.lookup(word.lower(), suggestion_verbosity, max_edit_distance_lookup)
          #Checks first uppercase to conserve the case.
          upperfirst = word[0].isupper()
          #Checks for correction suggestions.
          if len(suggestions) > 0:
              correction = suggestions[0].term
              replacement = correction
          #We call our _reduce_exaggerations function if no suggestion is found. Maybe there are repeated chars.
          else:
              replacement = _reduce_exaggerations(word)
          #Takes the case back to the word.
          if upperfirst:
              replacement = replacement[0].upper()+replacement[1:]
          word = replacement
          token_list[word_pos] = word
  return " ".join(token_list).strip()

  sentence_original="in te dhird qarter oflast jear he had elarned aoubt namials"
  sentence_corrected = spell_correction_text(sentence_original,
                                            spellchecker, 
                                            max_edit_distance_dictionary= 10,
                                            prefix_length = 1)
  print("Original:  " + sentence_original)
  print("Corrected: " + sentence_corrected)

####################################################################################
# NLP Pipeline for input text
####################################################################################

from spacy.lookups import Lookups
from spacy.lemmatizer import Lemmatizer

def removeEntityAnnotation(pattern):
  print('\nRemoving entity annotation for: ', pattern)
  #space or space and word
  pattern = re.sub(",( | \w+)\]", "", pattern)
  pattern = pattern.replace('[', "")
  print('Removed entity annotation: ', pattern)
  return pattern

#private
# Lemmatization function
def lemmatize(sentence_list, nlp):
    new_norm=[]
    print("Lemmatizing Sentences")
    for sentence in tqdm(sentence_list):
        new_norm.append(lemmatize_text(sentence, nlp).strip())
    return new_norm

#private
# Lemmatization is language dependent hence we need to pass Spacy "nlp" object 
def lemmatize_text(sentence, nlp):
    sent = ""
    doc = nlp(sentence)
    for token in doc:
        if '@' in token.text:
            sent+=" @MENTION"
        elif '#' in token.text:
            sent+= " #HASHTAG"
        else:
            sent+=" "+token.lemma_
    return sent


# Order? Remove stop words before lemmatizing maybe?
def nlpPipelineSentence(sentence):
  from spacy.lang.en.stop_words import STOP_WORDS
  import spacy
  import re
  nlp = spacy.load("en_core_web_sm")
  tokens_filtered = []
  tokens_stemmed = []
  tokens_noduplicates = []

  # Remove entity annotation from pattern
  #sentence = re.sub("([,]).*?([\]])", "", sentence)
  #sentence = sentence.replace('[', "")
  sentence = removeEntityAnnotation(sentence)
  print('Pattern without entity annotation:' , sentence, sep='')

  # Spelling correction
  sentence_spelling = spell_correction_text(sentence, spellchecker, max_edit_distance_dictionary= 10, prefix_length = 1)
  print('Spelling corrected sentence:' , sentence_spelling, sep='') 
  # Lemmatize
  sentence_lemmas = lemmatize_text(sentence_spelling, nlp)
  print('Lemmatized sentence:' , sentence_lemmas, sep='')
  # Tokenize, Remove puncuation, Remove stop words
  doc = nlp(sentence_lemmas)
  for token in doc:
    if not token.is_punct and token not in STOP_WORDS and token.text != ' ':
      tokens_filtered.append(token.text)
  print('Filtered tokens: ', tokens_filtered)
  # Stem (Snowball Stemmer)
  from nltk.stem.snowball import SnowballStemmer
  stemmer = SnowballStemmer(language='english')
  for token in tokens_filtered:
    tokens_stemmed.append(stemmer.stem(token))
  print('Stemmed tokens: ', tokens_stemmed)
  # Remove duplicates
  tokens_noduplicates = tokens_stemmed
  tokens_noduplicates = sorted(list(set(tokens_noduplicates)))
  print('Unduplicated tokens: ', tokens_noduplicates)
  return tokens_noduplicates


def nlpPipelineIntents(intents):
  import itertools
  classes = []
  words = []
  documents = []
  
  for intent in intents['intents']:
    for pattern in intent['patterns']:
      print('Run NLP Pipeline for pattern: ', pattern)
      #Classes
      if intent['tag'] not in classes:
        classes.append(intent['tag'])
      tokens = nlpPipelineSentence(pattern)
      print('Tokens: ', tokens)
      #Words
      words.append(tokens)
      #Documents
      documents.append((tokens, intent['tag']))

  #words = list(itertools.chain.from_iterable(words))
  words = list(itertools.chain(*words))
  print('\nClasses:',classes)
  print('Words:',words)
  print('Documents:',documents)
  return classes, words, documents

###################################################################################
# Let's define two needy functions to do the natural language preprocessing for us
# and build the bag of words (bow) for us from a sentence of words
###################################################################################

#private
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#private
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


def train_intents(words, classes, documents):
  global model

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



  ###########
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




  ##########
  print('These are all the words for our classification:\n\n', words)
  print('')
  print('Our sentence is represented by the following bag of words (bow):\n')
  p = bow("Can you please tell me if you are open today?", words, show_details=True)

  ##########

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

####################################################################################
# Named Entity Recognition
####################################################################################
import random

# specify the training data and the number of training iteration
def train_ner(data, iterations = 40, nlp = spacy.blank('en')): # creates blank Language class
    # spacy.blank('en') creates blank Language class
    training_data = data
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        print("Create new NER pipe...")
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        print("Get exiting NER pipe...")
        ner = nlp.get_pipe("ner")   

    # add labels
    for _, annotations in training_data:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(training_data)
            losses = {}
            for text, annotations in training_data:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.35,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp

def find_entities_per_ner(unmodified_sentence):
  print("\nFinding entities for ", unmodified_sentence)
  doc = nlp(unmodified_sentence)
  for ent in doc.ents:
    print('NER found entity for sentence <', unmodified_sentence, '>: ', ent.text, ent.start_char, ent.end_char, ent.label_)
  return doc.ents

######################################################################
# OLD NLP for neural network intent training
######################################################################

def old_nlp():
  words = []
  classes = []
  documents = []
  ignore_words = ['?']

  import re

  # loop through each sentence in our intents patterns
  for intent in intents['intents']:
    for pattern in intent['patterns']:

      # Remove entity annotation from pattern
      pattern = re.sub("([,]).*?([\]])", "", pattern)
      pattern = pattern.replace('[', "")
      #print('Pattern without entity annotation: ', pattern)
      
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

# Unused
########################################################################
# Extract Entities from gold format
# Example
# Input:  'Put the [plum, fruit] onto the [bed, place]'
# Output: ('[fruit, place]'}
########################################################################


#TRAIN_SINGLE = [('Put the plum onto the bed', {'entities': [(8, 12, 'fruit'), (22, 25, 'place')]})]

def getEntitiesFromTrainingData(training_data):
  entities = {}
  entityValues = [ ]

  for text, annotations in training_data:
    for anno in annotations['entities']:
      #print('Annotation: ', anno)
      startIndex = anno[0]
      endIndex = anno[1]
      entityType = anno[2]
      entityValue = text[startIndex:endIndex]
      #print('Entity value: ', entityValue)
      #print('Adding ', entityValue, ' to entities')
      entities.setdefault(entityType, []).append(entityValue)
      
  #tokens_noduplicates = sorted(list(set(tokens_noduplicates)))
  for key in entities.keys():
    unduped = sorted(list(set(entities[key])))
    entities[key] = unduped

  return entities

    
#getEntitiesFromTrainingData(TRAIN_FULL)

########################################################################
# Retrain NER with a newly added entity
########################################################################

# deprecated because it's based on gold format
#def addEntityToTrainingSet(newEntity = 'shelf', entityType = 'place'):
#  label  = [((0, len(newEntity), entityType))]
#  training_line = newEntity, {"entities": label}
#  print('Training new entity: ', training_line, ' with entity: ', newEntity)
#  TRAIN_ISOLATED.append(training_line)
#  print(TRAIN_ISOLATED)
#addEntityToTrainingSet()

def addEntityToReplacements(intents, entityValue = 'brot', entityType = 'vegetable'):
  print('Adding, ', entityValue,'as a ', entityType)

  newValues = intents['replacements'][entityType]
  newValues.append(entityValue)
  print('Updated values: ', newValues)
  #newValues = sorted(list(set(newValues)))
  print('Removed duplicate values: ', newValues)
  intents['replacements'][entityType].append(newValues)
  return intents


# For NER use the original sentence
# For Intent recognition use the NLPd sentence
def train_all(original_intents):
  # generate usable intents
  annotated_intents = patternCombinator(original_intents)
  print('\nAnnotated:', *annotated_intents['intents'], sep='\n')

  # NLP Pipeline for Intents
  #pipelined = nlpPipelineSentence('Uhm, this might be a better exemplörry sentence using a new sentence with mice. What about an [apple, fruit] for you');
  #print('Pipelined sentence: ', pipelined, '\n')
  #print(nlpPipelineSentence('apple'))
  classesWordsDocuments = nlpPipelineIntents(annotated_intents)

  # Intent training with pipelined intents
  classes = classesWordsDocuments[0]
  words = classesWordsDocuments[1]
  documents = classesWordsDocuments[2]
  train_intents(words, classes, documents)

  # Pure gold-parsed training data in addition to the intents
  global NER_TRAINING_DATA
  NER_TRAINING_DATA = parseToGoldFormat(annotated_intents)
  print('NER Training data:', NER_TRAINING_DATA)
  # Train NER
  print('Execute Train NER for: ', NER_TRAINING_DATA)
  train_ner(NER_TRAINING_DATA, iterations = 40)

  return classesWordsDocuments

def fetch_intent_json():
  ######################################################################
  # Import our chat-bot intents file
  ######################################################################
  
  #data_url = "https://raw.githubusercontent.com/rupel190/voiceinterfaceprogramming/main/app/src/intents.json"
  #req = requests.get(data_url)
  #intent_json = req.text

  data_url = "./intents.json"
  intent_json = open(data_url, 'r').read()
  #print(intent_json)

  intents = json.loads(intent_json)
  return intents

# classifies our sentence to a class
def classify(sentence):
    global classes
    global words
    ERROR_THRESHOLD=0.0001
    
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
  
def entities_to_required_entities(requiredEntities, sentence = "Put the banana onto the table"):
  entities = find_entities_per_ner(sentence)
  print('\nFound entities: ', entities)
  for found in entities:
    #override multiple entities
    entity_text = found.text
    entity_class = classify(entity_text)[0][0]
    requiredEntities[entity_class] = entity_text
    print("found: ", entity_text, "of kind: ", entity_class)
    print('Entity Dictionary: ', requiredEntities)

# fill at least 3 required entitites
def prompt(original_sentence):
  global remainingPrompts
  global requiredEntities
  # Check for required entities
  while remainingPrompts > 0:
    # NRE with original structure bc it uses that for training
    entities_to_required_entities(requiredEntities, original_sentence)
    #testGatherEntities()
    missingValues = []
    for key, value in requiredEntities.items():
      if not value:
        missingValues.append(key)
    remainingPrompts -= 1
    
    # enough
    if sum(map(len, requiredEntities.values())) >= 3:
      remainingPrompts = 3
      return requiredEntities
    return 'Please provide at least 3 values from these: ' + ', '.join(missingValues) + '. Remaining prompts: ' + str(remainingPrompts)
  # recipe for bugs if broken out from the prompt for another reason
  remainingPrompts = 3
  print('Continuing with following entities: ', requiredEntities)


def parseLearningInputSentence(original_sentence):
  # arbitrary text entityValue is a entityType arbitrary text
  expression = '(\w*) is a (\w*)'
  for match in re.finditer(expression, original_sentence):
    entityValue = match.group(1)
    entityType = match.group(2)
  return(entityValue, entityType)

def matchedIntent(intent):
  print('\nMatched intent:', intent)
  tag = intent['tag']

  if tag == 'fruit' or tag == 'vegetable':
    print('Fruit match')
    return prompt(original_sentence)
  if tag == 'listEntities':
    print('listEntities')
    return getEntitiesFromTrainingData(NER_TRAINING_DATA)
  if tag == 'learning':
    # no multithreading, no knowledge about ai, retrain everything
    newEntity = parseLearningInputSentence()
    entityValue = newEntity[0]
    entityType = newEntity[1]
    print('Learning with value', entityValue, 'and type', entityType)
    addEntityToReplacements(intents, entityValue, entityType)
    return 'Learning is deactivated because it would take too long but the logic is there.'
    #classesWordsDocuments = train_all(original_intents)
  else:
    return (random.choice(intent['responses']))



# generates a contextualized response for a specific user based on 3 elements
#       (a) class with highest prediction propability
#       (b) a specific user id
#       (c) context set
def response(sentence, user_id='123', show_details=True, resp='Beg your pardon?'):
  global context
  # Intents use NLPd sentence
  pipelined = ' '.join(nlpPipelineSentence(sentence))
  # TEST
  #pipelined = 'box in kiwi pleas put that the would'
  print('Pipelined intent:', pipelined)
  results = classify(pipelined)
  print('Classification: ', results)

  intent_copy = fetch_intent_json()
  # if we have a classification then find the matching intent tag
  if results:
    # loop as long as there are matches to process
    while results:
      for i in intent_copy['intents']:  
        # find a tag matching the first result
        if i['tag'] == results[0][0]:
          # in the matched intent now, e.g. fruit
          print('tag for intent: ', results[0][0])
          
          if user_id in context: 
            print('Current context:', context[user_id]) 
          else: 
            print('No context')

          if not 'context_filter' in i or (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
              print ('tag:', i['tag'])
              # pass the matched intent and return a random response from the intent
              resp = matchedIntent(i)

          # context_set
          if 'context_set' in i:
              print('Setting context:', i['context_set'])
              context[user_id] = i['context_set']  

          #print('Current response will be:', resp)
          return resp
      results.pop(0)
  return resp #


#######################################
# Globals for response
#######################################
# Hold and track the user context
context = {}
# Allow prompting for missing entities
remainingPrompts = 3
requiredEntities = {
  "fruit": "",
  "vegetable": "",
  "directions": "",
  "places": ""
}
#(context for fruit and veggies is called fruitContext currently)

#######################################
# DANGERZONE
#######################################

# All training
original_intents = fetch_intent_json()
print('Original intents:', original_intents)

classesWordsDocuments = train_all(original_intents)
print('Model:', model)

#classes = classesWordsDocuments[0]
#words = classesWordsDocuments[1]
#documents = classesWordsDocuments[2]
#print('Pipelined Intents Classes: ', classesWordsDocuments[0])
#print('Pipelined Intents Words: ', classesWordsDocuments[1])
#print('Pipelined Intents Documents: ', classesWordsDocuments[2])

#######################################
# Testrun
#######################################
#original_sentence = 'Hello'
#response(original_sentence)

#original_sentence = "give me a bunch of bananas to put it below"
#response(original_sentence)





