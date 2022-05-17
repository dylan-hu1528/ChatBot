# Import packages
import numpy as np
import pandas as pd

# Training data
train_df = pd.read_csv("covid_simple_training_data.csv")  # data frame of train data

train_df.columns = ['ID', 'fever', 'cough', 'breating-issue', 'infected']
pd.set_option("display.max_columns", 500) # Load all columns

# train_df.head() # display train data frame

# Test data
# test_df = pd.read_csv(test_data_url)  # data frame of test data

# train_df.columns = ['ID', 'fever', 'cough', 'breating-issue', 'infected']
# pd.set_option("display.max_columns", 500) # Load all columns

# test_df.head()


data = train_df.values  #2-D array of train data

def check_unique(data):
  class_col = data[:, 0]  # the whole column of the symptoms
  unique_classes = np.unique(class_col)

  if len(unique_classes) == 1:
    return True
  else:
    return False


def check_type(data):
  class_col = data[:, 0]
  unique_classes, counts_unique_classes = np.unique(class_col, return_counts=True)

  index = counts_unique_classes.argmax()
  type = unique_classes[index]

  return type

def get_possible_splits(data):
  possible_splits = {}
  _, n_cols = data.shape
  for col_index in range(1, n_cols):  
    vals = data[:, col_index]
    unique_vals = np.unique(vals)

    possible_splits[col_index] = unique_vals
    
  return possible_splits

def split_data(data, split_col, split_val):
  split_col_vals = data[:, split_col]
  
  data_below = data[split_col_vals == split_val]
  data_above = data[split_col_vals != split_val]

  return data_below, data_above

def calculate_entropy(data):
  class_col = data[:, 0] 
  _, counts = np.unique(class_col, return_counts=True)

  probabilities = counts / counts.sum()
  entropy = sum(probabilities * -np.log2(probabilities))

  return entropy


def calculate_overall_entropy(data_below, data_above):
  n = len(data_below) + len(data_above)

  p_data_below = len(data_below) / n
  p_data_above = len(data_above) / n

  overall_entropy = (p_data_below * calculate_entropy(data_below)) + (p_data_above * calculate_entropy(data_above))

  return overall_entropy

# find the lowest overall entropy
def find_best_split(data, possible_splits):
  overall_entropy = 9999
  for col_index in possible_splits:
    for val in possible_splits[col_index]:
      data_below, data_above = split_data(data, split_col=col_index, split_val=val)
      current_overall_entropy = calculate_overall_entropy(data_below, data_above)

      if current_overall_entropy <= overall_entropy:
        overall_entropy = current_overall_entropy
        best_split_col = col_index
        best_split_val = val
      
  return best_split_col, best_split_val

def decision_tree_algorithm(df, counter=0, min_samples=2):
  # get features
  if counter == 0:
    global COL_HEADERS
    COL_HEADERS = df.columns
    data = df.values
  else:
    data = df

  # base case
  if (check_unique(data)) or (len(data) < min_samples):
    type = check_type(data)
    return type

  # recursive part
  else:
   counter += 1
   # helper functions
   possible_splits = get_possible_splits(data)
   split_col, split_val = find_best_split(data, possible_splits)
   data_below, data_above = split_data(data, split_col, split_val)
   
   # build sub tree
   feature_name = COL_HEADERS[split_col]
   question = "{} = {}".format(feature_name, split_val)
   sub_tree = {question: []}

   # find answer
   yes_answer = decision_tree_algorithm(data_below, counter, min_samples)
   no_answer = decision_tree_algorithm(data_above, counter, min_samples)

   # if the answer is the same, do not ask question
   if yes_answer == no_answer:
     sub_tree = yes_answer
   else:
     sub_tree[question].append(yes_answer)
     sub_tree[question].append(no_answer)

  return sub_tree

def classify_example(example, tree):
  question = list(tree.keys())[0]
  feature_name, comparison_symbol, value = question.split()

  # ask question
  if example[feature_name] == value:
    answer = tree[question][0]
  else:
    answer = tree[question][1]

  # base case
  if not isinstance(answer, dict):
    return answer

  # recursive part
  else:
    rest_tree = answer
    return classify_example(example, rest_tree)

def calculate_accuracy(df, tree):
  df["type"] = df.apply(classify_example, axis=1, args=(tree,))
  df["type_correct"] = df.type == df.classes

  accuracy = df.type_correct.mean()

  return accuracy

