import codecs
import numpy as np
import jieba
import h5py,shelve
'''
f = h5py.File('wordData50.hdf5','r')
X = f['X'].value
Y = f['Y'].value
d = shelve.open('wordData50.data')
Vocab = d['Vocab']
id2c = d['id2c']
c2id = d['c2id']
#print(X,Y,Vocab,id2c,c2id)
'''
class Data():
    def __init__(self,datafilepath):
        file = codecs.open(datafilepath,'r').readlines()[1:]
        file = [i.replace('\n','') for i in file]
        files = [i.split(',') for i in file]
        self.id,self.X,self.Y = [],[],[]
        for idx,i in enumerate(file):
            self.id.append(files[idx][0])
            idlen = len(files[idx][0])
            self.X.append(i[idlen+1:])
        print(self.id[0],self.X[0])
    
    def makechar(self,padlen=50):
        d = shelve.open('charData50.data')
        Vocab = d['Vocab']
        id2c = d['id2c']
        c2id = d['c2id']
        charX = []
        for i in self.X:
            tmpline = []
            for j in i:
                tmpid = 0
                try:
                    tmpid = c2id[j]
                except Exception:
                    pass
                tmpline.append(tmpid)
            pad = padlen-len(tmpline)
            for k in range(0,pad):
                tmpline.append(0)
            tmpline = tmpline[0:padlen]
            charX.append(np.array(tmpline))
        charX = np.array(charX)
        return charX

    def savechar(self,padlen):
        charX = self.makechar()
        f = h5py.File("charTest"+str(padlen)+".hdf5","w")
        f.create_dataset('X',data=charX)
        f.close()        
        d = shelve.open("idTest"+str(padlen))
        d['id'] = self.id
        d.close()
        print(len(self.X),len(self.id))
    
    def getstopword(self):
        stopwords = set(['的','我'])
        return stopwords

    def worddata(self,padlen=50):
        stopwords = self.getstopword()
        splitX = []
        for i in self.X:
            seg = jieba.cut(i)
            seg = ','.join(seg)
            seg = seg.replace(' ','')
            seg = seg.split(',')
            splitX.append(seg)
        #count word number
        wordVocab = set()
        for i in splitX:
            for j in i:
                wordVocab.add(j)
        wordVocabSize = len(wordVocab)
        #complete dict for id and word
        id2word = {}
        word2id = {}
        for idx,i in enumerate(wordVocab):
            id2word[idx+1] = i
            word2id[i] = idx+1
        #make
        wordX = []
        wordY = []
        for i in splitX:
            tmpline = []
            for j in i:
                tmpid = word2id[j]
                tmpline.append(tmpid)
            pad = padlen-len(tmpline)
            for k in range(0,pad):
                tmpline.append(0)
            tmpline = tmpline[0:padlen]
            wordX.append(np.array(tmpline))
        wordX = np.array(wordX)
        wordX.astype(np.int32)
        wordY = [int(i) for i in self.Y]
        wordY = np.array(wordY)
        wordY = np.reshape(wordY,(-1,))
        return wordVocab,id2word,word2id,wordX,wordY

    def saveword(self,padlen):
        wordVocab,id2word,word2id,wordX,wordY = self.worddata()
        print('wordVocab:',len(wordVocab),'sampleX:',len(wordX))
        #print(wordVocab,id2word,word2id,wordY,wordX)
        f = h5py.File("wordData"+str(padlen)+".hdf5","w")
        f.create_dataset('X',data=wordX)
        f.create_dataset('Y',data=wordY)
        d = shelve.open("wordData"+str(padlen)+".data")
        d['Vocab'] = wordVocab
        d['id2c'] = id2word
        d['c2id'] = word2id
        f.close()
    
    def save(self,padlen=50):
        print('start....')
        self.savechar(padlen=padlen)
        print('save char done....')
        #self.saveword(padlen=padlen)
        #print('save word done....')

data = Data('YNU/predict_first.csv')
data.save()

#test:
'''
f = h5py.File('wordData50.hdf5','r')
X = f['X'].value
Y = f['Y'].value
d = shelve.open('wordData50.data')
Vocab = d['Vocab']
id2c = d['id2c']
c2id = d['c2id']
#print(X,Y,Vocab,id2c,c2id)
'''