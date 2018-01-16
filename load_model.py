from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Conv1D
from keras.models import load_model
import pickle

def load_cnn_imdb():
    model = load_model('weights/model_imdb.h5')
    print("Definizione del modello terminata")
    model.load_weights('weights/w_imdb.h5')
    print("Caricamento dei pesi terminato")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print("Compile del modello terminato")
    return model

def load_cnn_imdb_labels():
    return {'0': 'neg:             ',
            '1': 'pos:             '}

def load_imdb_word_index():
    with open('./word_index.pkl', 'rb') as f:
        return pickle.load(f)    
    
def load_imdb_keywords():
    with open('./keywords_tfidf.pkl', 'rb') as f:
        return pickle.load(f)    
    