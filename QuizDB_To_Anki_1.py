# import necessary modules
import pandas as pd
import numpy as np
import nltk.data
# initialize file paths
data_file_path  = "insert path to the csv file downloaded straight from QuizDB"
output_file_path = "insert path to the location where you want to save the csv file for Anki"

# download the sentence fragmenting package from the natural language processing toolkit
nltk.download('punkt')
data = pd.read_csv(data_file_path, encoding='utf-8')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
QAArr = []
# for loop to iterate through the questions and answers given by quizdb csv
for num in range(len(data.index)):
    dataForALgo = data.iloc[num]['Text'] # get the test for the question
    qArray = [token for token in tokenizer.tokenize(dataForALgo.strip())] # split the question into its sentences
    answerArray = [data.iloc[num]['Answer'].partition(' &l')[0] for int in range(len(qArray))] # duplicate the answers so there is an answer for each sentence 
    # turn python lists into numpy arrays
    qArray = np.asarray(qArray)
    answerArray = np.asarray(answerArray)
    extendArray = np.vstack((qArray, answerArray)) # concatenate question and answer arrays
    # add question-answer array to array that will be written to a csv file
    if len(QAArr) == 0:
        QAArr = extendArray
    else:
        QAArr = np.hstack((QAArr, extendArray))
# save as csv to the output file path location
df = pd.DataFrame(QAArr.T)
df.to_csv(output_file_path, header = None, index = None, encoding = 'utf-8-sig')