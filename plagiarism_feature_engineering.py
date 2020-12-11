#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Run in conda_amazonei_mxnet_p36 kernel

#!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip
#!unzip data

# import libraries
import pandas as pd
import numpy as np
import os

csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
plagiarism_df.head()

# Read in a csv file and return a transformed dataframe

def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things: 
       1) converts `Category` column values to numerical values 
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''
    # Read csv file
    df = pd.read_csv(csv_file)
    
    # Add a class column
    df["Class"] = df["Category"].replace({'non':0, 'heavy':1, 'light':1, 'cut':1, 'orig':-1})
    
    # Convert categoricals to numericals
    df["Category"] = df["Category"].replace({'non':0, 'heavy':1, 'light':2, 'cut':3, 'orig':-1})

    return df
    
# create new `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')
transformed_df.head(10)

# importing tests
import plagarism_unittests as tests

# test numerical_dataframe function
tests.test_numerical_df(numerical_dataframe)

# if above test is passed, create NEW `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
print('\nExample data: ')
transformed_df.head()

import plagarism_helper 

# create a text column 
text_df = plagarism_helper.create_text_column(transformed_df)
text_df.head()

# after running the cell above check out the processed text for a single file, by row index
row_idx = 0 # feel free to change this index
sample_text = text_df.iloc[0]['Text']
print('Sample processed text:\n\n', sample_text)

random_seed = 1 # can change; set for reproducibility

# create new df with Datatype (train, test, orig) column
# pass in `text_df` from above to create a complete dataframe, with all the information you need
complete_df = plagarism_helper.train_test_dataframe(text_df, random_seed=random_seed)

# check results
complete_df.head(10)

from sklearn.feature_extraction.text import CountVectorizer

# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text, 
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''
    
    # get the text for the answer and the source
    answer_text, answer_task  = df[df.File == answer_filename][['Text', 'Task']].iloc[0]
    source_text = df[(df.Task == answer_task) & (df.Class == -1)]['Text'].iloc[0]
    
    # count the n-grams
    counter = CountVectorizer(analyzer='word', ngram_range=(n,n))
    ngrams_array = counter.fit_transform([answer_text, source_text]).toarray()
        
    ## Calculate containment 
    count_common_ngrams = sum(min(a, s) for a, s in zip(*ngrams_array))
    count_ngrams_a = ngrams_array[0].sum()
    normalize = count_common_ngrams / count_ngrams_a
    
   # intersection = np.arange(len(ngrams_array[0]))[ngrams_array[0]==ngrams_array[1]]
   # common_terms = len(intersection)
   # normalize = common_terms/(np.sum(ngrams_array[0]))
    
    return normalize

# select a value for n
n = 3

# indices for first few files
test_indices = range(5)

# iterate through files and calculate containment
category_vals = []
containment_vals = []
for i in test_indices:
    # get level of plagiarism for a given file index
    category_vals.append(complete_df.loc[i, 'Category'])
    # calculate containment for given file and n
    filename = complete_df.loc[i, 'File']
    c = calculate_containment(complete_df, n, filename)
    containment_vals.append(c)

# print out result, does it make sense?
print('Original category values: \n', category_vals)
print(str(n)+'-gram containment values: \n', containment_vals)

# test containment calculation
# params: complete_df from before, and containment function
tests.test_containment(complete_df, calculate_containment)

# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''
    
    answer_words_list, source_words_list = answer_text.split(), source_text.split()
    answer_word_count, source_word_count = len(answer_words_list), len(source_words_list)

    lcs_matrix = np.zeros((source_word_count + 1, answer_word_count + 1), dtype=int)

    for source, source_word in enumerate(source_words_list, 1):
        for answer, answer_word in enumerate(answer_words_list, 1):
            if source_word == answer_word:
                lcs_matrix[source][answer] = lcs_matrix[source-1][answer-1] + 1
            else:
                lcs_matrix[source][answer] = max(lcs_matrix[source-1][answer], lcs_matrix[source][answer-1])
    
    matrix = lcs_matrix[source_word_count][answer_word_count]
    
    return matrix / answer_word_count

A = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents"
S = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents"

# calculate LCS
lcs = lcs_norm_word(A, S)
print('LCS = ', lcs)

# expected value test
assert lcs==20/27., "Incorrect LCS value, expected about 0.7408, got "+str(lcs)

print('Test passed!')

# test lcs implementation
# params: complete_df from before, and lcs_norm_word function
tests.test_lcs(complete_df, lcs_norm_word)

# test on your own
test_indices = range(5) # look at first few files

category_vals = []
lcs_norm_vals = []
# iterate through first few docs and calculate LCS
for i in test_indices:
    category_vals.append(complete_df.loc[i, 'Category'])
    # get texts to compare
    answer_text = complete_df.loc[i, 'Text'] 
    task = complete_df.loc[i, 'Task']
    # we know that source texts have Class = -1
    orig_rows = complete_df[(complete_df['Class'] == -1)]
    orig_row = orig_rows[(orig_rows['Task'] == task)]
    source_text = orig_row['Text'].values[0]
    
    # calculate lcs
    lcs_val = lcs_norm_word(answer_text, source_text)
    lcs_norm_vals.append(lcs_val)

# print out result, does it make sense?
print('Original category values: \n', category_vals)
print('Normalized LCS values: \n', lcs_norm_vals)

# Function returns a list of containment features, calculated for a given n 
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):
    
    containment_values = []
    
    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n
    
    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i,'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks 
        else:
            containment_values.append(-1)
    
    print(str(n)+'-gram containment features created!')
    return containment_values

# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):
    
    lcs_values = []
    
    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature for answer tasks
        if df.loc[i,'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text'] 
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks 
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values
    
# Define an ngram range
ngram_range = range(1,21)

features_list = []

# Create features in a features_df
all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

# Calculate features for containment for ngrams in range
i=0
for n in ngram_range:
    column_name = 'c_'+str(n)
    features_list.append(column_name)
    # create containment features
    all_features[i]=np.squeeze(create_containment_features(complete_df, n))
    i+=1

# Calculate features for LCS_Norm Words 
features_list.append('lcs_word')
all_features[i]= np.squeeze(create_lcs_features(complete_df))

# create a features dataframe
features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

# Print all features/columns
print('Features: ', features_list)

# print some results 
features_df.head(10)

# Create correlation matrix for just Features to determine different models to test
corr_matrix = features_df.corr().abs().round(2)

# display shows all of a dataframe
####display(corr_matrix)

# Takes in dataframes and a list of selected features (column names) 
# and returns (train_x, train_y), (test_x, test_y)
def train_test_data(complete_df, features_df, selected_features):
    '''Gets selected training and test features from given dataframes, and 
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''
    
    df = pd.concat([complete_df, features_df[selected_features]], axis=1)    
    df_train, df_test = df[df.Datatype == 'train'], df[df.Datatype == 'test']

    # get the training features and labels
    train_x, train_y = df_train[selected_features].values, df_train['Class'].values
    
    # get the test features and labels
    test_x, test_y = df_test[selected_features].values, df_test['Class'].values
    
    return (train_x, train_y), (test_x, test_y)


test_selection = list(features_df)[:2] # first couple columns as a test
# test that the correct train/test data is created
(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, test_selection)

# params: generated train/test data
tests.test_data_split(train_x, train_y, test_x, test_y)

# Select list of features, this should be column names from features_df
# ex. ['c_1', 'lcs_word']

selected_features = ['c_1', 'c_11', 'lcs_word']
(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, selected_features)

# check that division of samples seems correct
# these should add up to 95 (100 - 5 original files)
print('Training size: ', len(train_x))
print('Test size: ', len(test_x))
print('Training df sample: \n', train_x[:10])

def make_csv(x, y, filename, data_dir):
    '''Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       '''
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # first column is the labels and rest is features 
    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv(os.path.join(data_dir, filename), header=False, index=False)
    
    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: '+str(data_dir)+'/'+str(filename))
    
fake_x = [ [0.39814815, 0.0001, 0.19178082], 
           [0.86936937, 0.44954128, 0.84649123], 
           [0.44086022, 0., 0.22395833] ]

fake_y = [0, 1, 1]

make_csv(fake_x, fake_y, filename='to_delete.csv', data_dir='test_csv')

# read in and test dimensions
fake_df = pd.read_csv('test_csv/to_delete.csv', header=None)

# check shape
assert fake_df.shape==(3, 4), \
      'The file should have as many rows as data_points and as many columns as features+1 (for indices).'
# check that first column = labels
assert np.all(fake_df.iloc[:,0].values==fake_y), 'First column is not equal to the labels, fake_y.'
print('Tests passed!')

# can change directory, if you want
data_dir = 'plagiarism_data'
#! rm -rf test_csv
make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)
make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)
