import pandas as pd
import numpy as np
import nltk.data
data_file_path  = "insert path to the csv file downloaded straight from QuizDB"
output_file_path = "insert path to the location where you want to save the csv file for Anki"

nltk.download('punkt')
data = pd.read_csv(data_file_path)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
QAArr = []
for num in range(len(data.index)):
    dataForALgo = data.iloc[num]['Text']
    qArray = [token for token in tokenizer.tokenize(dataForALgo.strip())]
    answerArray = [data.iloc[num]['Answer'].partition(' &l')[0] for int in range(len(qArray))]
    qArray = np.asarray(qArray)
    answerArray = np.asarray(answerArray)

    extendArray = np.vstack((qArray, answerArray))
    if len(QAArr) == 0:
        QAArr = extendArray
    else:
        QAArr = np.hstack((QAArr, extendArray))
df = pd.DataFrame(QAArr.T)
df.to_csv(output_file_path, header = None, index = None)