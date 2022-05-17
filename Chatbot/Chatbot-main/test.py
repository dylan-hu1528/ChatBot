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


fever = input("Do you have a fever? (Yes or No) ")
cough = input("Do you cough? (Yes or No) ")
breathing_issue = input("Do you have short breating or other breathing issues? (Yes or No) ")
infected = "Yes"
test_sample = fever + "," + cough + "," + breathing_issue + "," +infected
f = open("test.txt", "w")
f.write(test_sample)
# convert to .csv
test_df = pd.read_csv(r'/Users/zhuofanli/Desktop/3521Chatbot/test.txt', header=None, delim_whitespace=True)  # data frame of test data
train_df.columns = ['fever', 'cough', 'breating-issue', 'infected']
pd.set_option("display.max_columns", 500) # Load all columns

test_df.head()