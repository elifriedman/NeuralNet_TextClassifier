from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing import text
import numpy as np
import random
import sys


corpus_num = 1
print "Using corpus #%d" % corpus_num
max_words = 30000
trainfrac = 0.8
batch_size = 32
nb_epoch = 5

def read_corpus(filename):
    trainfile = open(filename)
    ind = filename.rfind('/')
    prefix = filename[:ind+1] if ind > -1 else ""
    texts,classes = [],[]
    for line in trainfile:
        line = line.strip()
        name,clss = line.split(' ')
        texts.append(open(prefix+name).read())
        classes.append(clss)
    return texts,classes
    
def traintest_split(X,y,frac):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    trainnum = int(frac*len(y))
    testnum = len(y)-trainnum
    return X[idx[:trainnum],:],y[idx[:trainnum],:],X[idx[trainnum:],:],y[idx[trainnum:],:]

filename = 'TC_provided/corpus'+str(corpus_num)+'_train.labels'
texts,classes = read_corpus(filename)
nb_classes = len(set(classes))

tokenizer = text.Tokenizer(nb_words=max_words)
tokenizer.fit_on_texts(texts)
print "Tokenizer has %d words" % len(tokenizer.word_index)

X = tokenizer.texts_to_matrix(texts,mode='tfidf')

class_to_idx = dict((clss,i) for i,clss in enumerate(set(classes)))
y = np.zeros((len(classes),len(class_to_idx)))
for i,clss in enumerate(classes):
    y[i,class_to_idx[clss]] = 1.0

X_train,y_train, X_test,y_test = traintest_split(X,y,trainfrac)


print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(Dense(200, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print "Training..."
history = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])
