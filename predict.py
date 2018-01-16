from matplotlib import pyplot as plt
from keras.preprocessing import text
from keras import backend as K
import numpy as np#
import tensorflow as tf
import re, csv

K._LEARNING_PHASE = tf.constant(0)
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 600

def get_prediction (text, model, word_index):
    x = []
    text = re.sub('([.,!?()\"\'])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    for word in text.split(' '):
        word = word.lower()
        try:
            index = word_index[word]
            if index >= MAX_NB_WORDS:
                x.append(0)
            else:
                x.append(index)
        except:
            x.append(0)
    while len(x) < MAX_SEQUENCE_LENGTH:
        x.insert(0, 0)
    x = np.asmatrix(x)
    prediction = model.predict(x) 
    return prediction

def get_predictions(original_text, perturbated, filename, m):
    
    model = m[0]
    labels = m[1]
    word_index = m[2]
    
    out_file = open("results/predictions_"+filename+".txt", "w") 
    
    
    original_prediction = get_prediction(original_text, model, word_index)
    out_file.write('Original text: ' + original_text + ' \n\n')
    for i in range(0, len(labels)):
        percentage = float(int(original_prediction[0][i]*10000))/100
        #if percentage > 0.9:
        out_file.write(labels[str(i)]  + str(percentage) + ' %\n')
    
    good ={}
    bad = {}
    
    for p in perturbated:
        prediction = get_prediction(perturbated[p], model, word_index)
        delta= original_prediction - prediction
        ratio = original_prediction/prediction

        or_percentage = float(int(original_prediction[0][1]*10000))/100
        percentage = float(int(prediction[0][1]*10000))/100
        if prediction[0][1] != 0:
            r_value = float(int(ratio[0][1]*100))/100
        else:
            r_value = 99999

        if abs(delta[0][1]) > 0.01:
            if r_value > 1:
                good[(p+':').ljust(20, ' ') + str(percentage) + ' -> '+ str(or_percentage) + ' %'] = r_value
            elif r_value < 1:
                bad[(p+':').ljust(20, ' ') + str(percentage) + ' -> '+ str(or_percentage) + ' %'] = r_value
            #out_file.write('\n\n - - - - - - - - - - - - \n\n')
            #out_file.write('Adding feature \''+p+'\': '+ str(percentage) + ' -> '+ str(or_percentage) + ' % ')       

    goodlen = len(good)
    if goodlen != 0:
        out_file.write("\n\nRanking of words with positive influence:\n")
        
        while goodlen - len(good) < 100 and len(good) > 0:
            km = max(good, key=good.get)
    
            vm = good[km]
    
            good.pop(km, None)
            out_file.write(' ' + str(vm).ljust(4, ' ') + ' - '+ km + '\n')
    
    badlen = len(bad)
    if badlen != 0:   
        out_file.write("\nRanking of words with negative influence:\n")
        while badlen - len(bad) < 100 and len(bad) > 0:
            km = min(bad, key=bad.get)
    
            vm = bad[km]
    
            bad.pop(km, None)
            out_file.write(' ' + str(vm).ljust(4, ' ') + ' - '+ km + '\n')
        
    out_file.close()  
    print('Predizioni terminate')