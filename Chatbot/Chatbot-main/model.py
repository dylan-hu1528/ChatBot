# Import packages
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle
import pandas as pd

# Import functions
import id3

# Import data from json file
with open("data.json") as file:
  data = json.load(file)


# Open stored model
try:
  with open("data.pickle", "rb") as f:
    words, labels, training, training, output = pickle.load(f)
except:
  words = []  # a list of different words
  labels = [] # a list of tags
  docs_x = [] # a list of the words user enter
  docs_y = [] # a list of what a input is part of

  for intent in data["intents"]: # dictionaries in intents
    for _input in intent["inputs"]:  # user inputs in dictionary
      wrds = nltk.word_tokenize(_input) # return to a list with all different words
      words.extend(wrds)
      docs_x.append(wrds)
      docs_y.append(intent["tag"])

      if intent["tag"] not in labels:
        labels.append(intent["tag"])

  # Remove duplicates
  words = [stemmer.stem(w.lower()) for w in words if w != "?"]
  words = sorted(list(set(words)))  # set removes duplicate

  labels = sorted(labels)

  # Bag of words
  training = []
  output = []

  out_empty = [0 for _ in range(len(labels))]

  for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
      if w in wrds:
        bag.append(1)
      else:
        bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

  traning = numpy.array(training)
  output = numpy.array(output)

  # If open exit model fails, build new and restore
  with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, training, output), f)

# Train the model
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])  # Find input shape for our model
net = tflearn.fully_connected(net, 8) # 8 neurons for layer
net = tflearn.fully_connected(net, 8) # 8 neurons for hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # get probablities for each output, output layer
net = tflearn.regression(net)  


model = tflearn.DNN(net)

try:
  model.load("model.tflearn")
except: 
  model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
  model.save("model.tflearn")

# Predict
def bag_of_words(s, words): 
  bag = [0 for _ in range(len(words))]  # Store all the words

  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
    for i, w in enumerate(words):
      if w == se: 
        bag[i] = 1

  return numpy.array(bag)

# Chat function
def chat():
  print("Start talking with our bot (type quit to end conversation)!")
  while True:
    inp = input("You: ")
    if inp.lower() == "quit":
      break
    
    # User continue to talk, turn inp to bag of words
    results = model.predict([bag_of_words(inp, words)])

    # Print just probability, delete when submit project
    results_index = numpy.argmax(results) # Give the max posiblity in results list
    tag = labels[results_index]

    for word_tg in data["intents"]:
      if word_tg["tag"] == tag:
        responses = word_tg["responses"]
    
    print(random.choice(responses))


# Start the chatbot
print("Do you want to play with chatbot or self-check whether you need to do a COVID test? (P for play, C for check)")
answer = input("I want to: ")
if answer == 'P':
  chat()
else:
  # Ask for symptom
  fever = input("Do you have a fever? (Yes or No) ")
  cough = input("Do you cough? (Yes or No) ")
  breathing_issue = input("Do you have short breating or other breathing issues? (Yes or No) ")
  infected = "Yes"

  # Write to a file
  test_sample = "1, " + fever + "," + cough + "," + breathing_issue + "," +infected
  f = open("test.txt", "w")
  f.write(test_sample)
  # convert to .csv
  test_df = pd.read_csv(r'/Users/zhuofanli/Desktop/3521Chatbot/test.txt', header=None, delim_whitespace=True)  # data frame of test data
  train_df.columns = ['ID', 'fever', 'cough', 'breating-issue', 'infected']
  pd.set_option("display.max_columns", 500) # Load all columns

  # test_df.head()
    # Calculate accuracy
  id3.accuracy = calculate_accuracy(df, tree)
  if accuracy >= 0.5:
    print("There is a large chance that you got COVID-19.")
  else:
    print("There is a large chance that you did not get COVID-19.")


  