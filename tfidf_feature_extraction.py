import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle, re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


BASE_DIR = ''
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'imdb_dataset')
MAX_SEQUENCE_LENGTH = 600
MAX_NB_WORDS = 10000


print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):

        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            t = t.replace('\n', '')
            t = re.sub(r'[^\w\s]','',t)
            tokens = word_tokenize(t)
            #filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
            lmtzr = WordNetLemmatizer()
            lems = [lmtzr.lemmatize(t) for t in tokens]
            t = " ".join(lems)
            texts.append(t)
            f.close()
            labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
with open("word_index.pkl", "wb") as w_index_file:
    pickle.dump(word_index, w_index_file)
print('Found %s unique tokens.' % len(word_index))

keywords={}

mat = tokenizer.texts_to_matrix(texts, 'tfidf')
for i in range (0, 50001):
    print(i)
    for j in range(0, 8000):
        if(mat[i][j] > 8):
            for key, value in word_index.items():
                if(value == j and len(key) > 2):
                    keywords[key] = value


with open("keywords_tfidf.pkl", "wb") as keyword_file:
    pickle.dump(keywords, keyword_file)

out_file = open("keywords_tfidf.txt", "w")
out_file.write(str(keywords))
out_file.close()