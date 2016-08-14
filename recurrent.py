from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import text
from tools.utils import read_corpus,traintest_split
import numpy as np
import random
import sys

corpus_num = 1
print "Using corpus #%d" % corpus_num
max_words = 30000
trainfrac = 0.8
batch_size = 32
nb_epoch = 10
# np.random.seed(14)

filename = 'TC_provided/corpus'+str(corpus_num)+'_train.labels'
texts,classes = read_corpus(filename)
nb_classes = len(set(classes))

tokenizer = text.Tokenizer(nb_words=max_words)
tokenizer.fit_on_texts(texts)
nb_words = len(tokenizer.word_index)
print "Tokenizer has %d words" % nb_words

seqs = tokenizer.texts_to_sequences(texts)
maxlen = max([len(seq) for seq in seqs])
X = np.array([seq+[0]*(maxlen-len(seq)) for seq in seqs])

class_to_idx = dict((clss,i) for i,clss in enumerate(set(classes)))
y = np.zeros((len(classes),len(class_to_idx)))
for i,clss in enumerate(classes):
    y[i,class_to_idx[clss]] = 1.0

X_train,y_train, X_test,y_test = traintest_split(X,y,trainfrac)


print('Building model...')
model = Sequential()
model.add(Embedding(input_dim=tokenizer.nb_words,output_dim=400,input_length=X.shape[1],mask_zero=True))
model.add(LSTM(128,dropout_W=0.3,dropout_U=0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print "Training..."
history = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

history.history['test_loss'] = score[0]
history.history['test_acc'] = score[1]
history.history['corpus'] = corpus_num

fname = __file__
fname = fname[fname.rfind('/')+1:]
np.save('logs/'+fname+'_hist',history.history)
