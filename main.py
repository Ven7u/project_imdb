from predict import get_predictions
from load_model import load_cnn_imdb, load_cnn_imdb_labels, load_imdb_word_index, load_imdb_keywords
import os
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    
    model = load_cnn_imdb()
    labels = load_cnn_imdb_labels()
    word_index = load_imdb_word_index()
    features = load_imdb_keywords()
    
    
    m = [model, labels, word_index]

    for filename in os.listdir('reviews/'):
        filename = filename.split('.')[0]
        news_file = open("reviews/"+filename+".txt", "r", encoding="utf8")
        text = news_file.read().replace('\n', '')
        text = re.sub(r'[^\w\s]','',text)
        tokens = word_tokenize(text)
        filtered_tokens = [word.lower() for word in tokens if word not in stopwords.words('english')]
        lmtzr = WordNetLemmatizer()
        lems = [lmtzr.lemmatize(t) for t in filtered_tokens]

        
        instance_features = []
        
        for l in lems:
            for word in features.keys():
                if(l == word and l not in instance_features):
                    instance_features.append(word)
        
        perturbated = {}
        
        for f in instance_features:
            p = []
            for t in tokens:
                if(t != f):
                    p.append(t)
            perturbated[f] =  " ".join(p)
         
        original_text = " ".join(tokens)
        
        out_file = open("results\perturbated_" + filename + ".txt", "w")
        out_file.write(text + "\n\n")
        for p in perturbated:
            out_file.write(p + "\n" + perturbated[p] + "\n\n")
        out_file.close()

        get_predictions(original_text, perturbated, filename, m)
        
        #get_predictions(perturbated, features, text, filename, m)

        
if __name__ == "__main__":
    main()