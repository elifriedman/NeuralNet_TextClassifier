import numpy as np

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

def reverse_text(text):
    textlist = text.split(' ')
