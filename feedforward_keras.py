from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing import text
from tools.utils import read_corpus,split,getMaxSparsity
from tools.layer import Layer
import numpy as np
import random
import sys
import nltk


stemmer = nltk.PorterStemmer()
def nltkTokenize(texts):
    newTexts = []
    for text in texts:
        words = nltk.word_tokenize(text)
        newList = []
        for word in words:
            word = word.lower()
            word = stemmer.stem(word)
        newTexts.append(" ".join(words))
    return newTexts

randomSeed = np.random.randint(0,200)
print("random num for random seeder (we understand how rand works but we're doing this anyways",randomSeed)
np.random.seed(randomSeed)

corpus_num = 1
histories = []
for corpus_num in [1,2,3]:
    print "Using corpus #%d" % corpus_num
    max_words = 30000
    trainfrac = 0.8
    batch_size = 32
    nb_epoch = 20

    filename = 'TC_provided/corpus'+str(corpus_num)+'_train.labels'
    texts,classes = read_corpus(filename)
    nb_classes = len(set(classes))

    # split into train and test classes
    texts_tr, texts_te, idx = split(texts,trainfrac)
    class_tr, class_te, blah = split(classes,trainfrac,idx)

    #texts_tr = nltkTokenize(texts_tr)
    #texts_te = nltkTokenize(texts_te)

    tokenizer = text.Tokenizer(nb_words=max_words)
    tokenizer.fit_on_texts(texts_tr)
    print "Tokenizer has %d words" % len(tokenizer.word_index)

    X_train = tokenizer.texts_to_matrix(texts_tr,mode='tfidf')
    X_test = tokenizer.texts_to_matrix(texts_te,mode='tfidf')

    class_to_idx = dict((clss,i) for i,clss in enumerate(set(class_tr)))
    y_train = np.zeros((len(class_tr),len(class_to_idx)))
    for i,clss in enumerate(class_tr):
        y_train[i,class_to_idx[clss]] = 1.0

    class_to_idx = dict((clss,i) for i,clss in enumerate(set(class_te)))
    y_test = np.zeros((len(class_te),len(class_to_idx)))
    for i,clss in enumerate(class_te):
        y_test[i,class_to_idx[clss]] = 1.0

    sparsity = getMaxSparsity(X_train)
    print("Sparsity",sparsity)
    print('Building model...')
    model = Sequential()
    model.add(Dense(round(sparsity/2), input_shape=(max_words,)))
    model.add(Activation('tanh'))
    model.add(Dense(round(sparsity/3)))
    model.add(Activation('tanh'))
    model.add(Dense(round(sparsity/4)))
    model.add(Activation('tanh'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    print "Training..."
    history = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    history.history['test_loss'] = score[0]
    history.history['test_acc'] = score[1]
    history.history['corpus'] = corpus_num
    histories.append(history.history)

fname = __file__
fname = fname[fname.rfind('/')+1:]
np.save('logs/'+fname+'_hist',histories)
