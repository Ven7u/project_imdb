from __future__ import print_function

import os, sys, pickle, re, numpy, csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
import nltk, pickle
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

MAX_NB_WORDS = 5000
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'imdb_dataset')

if not os.path.exists('corpus.pkl'):    #il corpus è stato già processato e salvato
    with open('corpus.pkl', 'rb') as f:
        texts = pickle.load(f) 
else:                                   # il corpus non è stato ancora processato
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    
    
    print('Processing text dataset')
    
    texts = []  # list of text samples
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path) and name == "pos":
            for fname in sorted(os.listdir(path)):
                    print(fname)
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    t = t.replace('\n', '')
                    t = re.sub(r'[^\w\s]','',t)
                    tokens = word_tokenize(t)
                    #filtered_tokens = [word.lower() for word in tokens if word not in stop_words]
                    lmtzr = WordNetLemmatizer()
                    lems = [lmtzr.lemmatize(t) for t in tokens]
                    texts.append(" ".join(lems))
                    f.close()
    
    print('Found %s texts.' % len(texts))
    with open("corpus_pos.pkl", "wb") as corpus_file:
        pickle.dump(texts, corpus_file)

#------------------------------------------------------------------------------------------------------------#
    
with open('keywords_tfidf.pkl', 'rb') as f:
    keywords = pickle.load(f) 
    
count_vect = CountVectorizer(stop_words= "english")
X_train_counts = count_vect.fit_transform(texts)

vocabulary = count_vect.vocabulary_

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

svd = TruncatedSVD(n_components = 10)
svd_matrix_terms = svd.fit_transform(X_train_tf.T)
#svd_matrix_docs = svd.fit_transform(X_train_tf)

csvReport = open('svd.csv', 'w', newline='')
for v in vocabulary:
    if(v in keywords):  
        csvReport.write(v + ";")  
        for sv in svd_matrix_terms[vocabulary[v]]:            
            csvReport.write(str(sv) + ";")
        csvReport.write("\n")
csvReport.close()

#------------------------------------------------------------------------------------------------------------#
   
vcos = cosine_similarity( svd_matrix_terms[vocabulary["skywalker"]] ,svd_matrix_terms)
vlist = []
for v in vocabulary.keys():
    k = vocabulary[v]
    vlist.append((vcos[0][k], v))
    print(v)

csvReport = open('prove.csv', 'w', newline='')
for v in vlist:
    csvReport.write(str(v[0]) + ";" + str(v[1]) + "\n") 
csvReport.close() 
   
#------------------------------------------------------------------------------------------------------------#
