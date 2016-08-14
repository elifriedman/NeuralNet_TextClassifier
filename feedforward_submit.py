import theano
import theano.tensor as T
import numpy as np
from tools.tokenizer import Tokenizer
from tools.utils import read_corpus,traintest_split
from tools.layer import Layer
from tools.utils import split,getMaxSparsity
import nltk
import sys


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

 

def main():

    max_words = 30000
    batch_size = 32
    nb_epoch = 50
    randomSeed = np.random.randint(0,200)
#  print("random num for random seeder (we understand how rand works but we're doing this anyways",randomSeed)
    
  
    s = "Please enter the name of a training file: "
    filename = raw_input(s)
    try:
        texts,classes = read_corpus(filename)
    except:
        print "Couldn't open file %s" % filename
        sys.exit(1)

    nb_classes = len(set(classes))

    tokenizer = Tokenizer(nb_words=max_words)
    tokenizer.fit_on_texts(texts)
    print "Tokenizer has %d words" % len(tokenizer.word_index)

    X = tokenizer.texts_to_matrix(texts,mode='tfidf')
    X = X.astype(theano.config.floatX)

    idx_to_class = dict((i,clss) for i,clss in enumerate(set(classes)))
    class_to_idx = dict((clss,i) for i,clss in enumerate(set(classes)))
    y = np.zeros((len(classes),len(class_to_idx)),dtype=theano.config.floatX)
    for i,clss in enumerate(classes):
        y[i,class_to_idx[clss]] = 1.0

#    X_train,y_train, X_test,y_test = traintest_split(X,y,trainfrac)

    sparsity = getMaxSparsity(X)
#    print("Sparsity",sparsity)
    sizes = [sparsity/8,sparsity/12,sparsity/15]
    s = [size for size in sizes]
    s.insert(0,X.shape[1])
    s.append(nb_classes)
    print "Layer sizes: ",s
    rng = np.random.RandomState(44)

    train_x = theano.shared(X)
    train_y = theano.shared(y)

    index = T.lscalar()
    x = T.matrix('x')
    o = T.matrix('o')
    layers = []
    inp = x
    inpshp = X.shape[1]
    params = []

    for size in sizes:
        layers.append(Layer(rng,inp,n_in=inpshp,n_out=size,activation=T.tanh))
        inpshp = size
        inp = layers[-1].output
        params += layers[-1].params

    L2 = Layer(rng,layers[-1].output,layers[-1].n_out,nb_classes,activation=T.nnet.softmax)
    cost = T.sum(T.nnet.categorical_crossentropy(L2.output,o).flatten())
    accuracy = T.sum(T.eq(T.argmax(L2.output,axis=1),T.argmax(o,axis=1)))

    L2.train_function(cost,params,
                      inputs=[index],
                      outputs=[cost,accuracy,L2.output],
                      givens={
                        x: train_x[index * batch_size: (index + 1) * batch_size],
                        o: train_y[index * batch_size: (index + 1) * batch_size]
                      },
                      update_type='adam',
                      paramlist=[])
#    L2.test_function(inputs=[],outputs=[accuracy,cost,L2.output],givens={x: X_test, o: y_test})

    num_batches = len(X)/batch_size
    accuracies = 1.0*np.zeros((nb_epoch,num_batches))

    print "Training..."
    check = 5
    last_accs = [0.0 for i in range(check)]
    for i in range(nb_epoch):
        print "Epoch %d/%d" % (i,nb_epoch)
        for j in range(num_batches):
            blah,accuracies[i,j],blah2 = L2.train(j)
        a = accuracies[i,:].sum()/len(X)
        last_accs.append(a)
#        print "loss: %s - acc: %s" % (repr(c),repr(a))
        if abs(sum(last_accs[-5:])/check-a) < 0.000001:
          print "accuracy isn't changing, so break"
          break

    print "Done training"


    s = "Please enter the name of the testing file: "
    filename = raw_input(s)
    files = []

    try:
        testfile = open(filename)
        ind = filename.rfind('/')
        prefix = filename[:ind+1] if ind > -1 else ""
        texts = []
        for line in testfile:
            line = line.strip()
            name = line
            files.append(name)
            try:
                texts.append(open(prefix+name).read())
            except:
                print "Couldn't open %s" % (prefix+name)
                sys.exit(1)
    except:
        print "Couldn't open file %s" % filename
        sys.exit(1)

    print "Testing..."

    X_test = tokenizer.texts_to_matrix(texts,mode='tfidf')
    X_test = X_test.astype(theano.config.floatX)

    L2.test_function(inputs=[],outputs=[L2.output],givens={x: X_test})

    output = L2.test()[0]
    output = output.argmax(axis=1) 


    s = "Please enter the name of the output file: "
    filename = raw_input(s)
    f = open(filename,'w')

    for i,out in enumerate(output):
        f.write(files[i]+" "+idx_to_class[out]+'\n')


if __name__ == '__main__':
  main()
