import codecs
import numpy as np
import jieba
import h5py,shelve
class Data():
    def __init__(self,datafilepath):
        file = codecs.open(datafilepath,'r').readlines()
        file = [i.split(',') for i in file]
        self.id = [i[0] for i in file]
        self.X = [i[1] for i in file]
        self.Y = [i[2].replace('\n','') for i in file]
    
    def chardata(self,padlen=50):
        #count char number
        charVocab = set()
        for i in self.X:
            for j in i:
                charVocab.add(j)
        charVocabSize = len(charVocab)
        #complete dict for id and char
        id2char = {}
        char2id = {}
        for idx,i in enumerate(charVocab):
            id2char[idx+1] = i
            char2id[i] = idx+1
        #make
        charX = []
        charY = []
        for i in self.X:
            tmpline = []
            for j in i:
                tmpid = char2id[j]
                tmpline.append(tmpid)
            pad = padlen-len(tmpline)
            for k in range(0,pad):
                tmpline.append(0)
            tmpline = tmpline[0:padlen]
            charX.append(np.array(tmpline))
        charX = np.array(charX)
        charY = [int(i) for i in self.Y]
        charY = np.array(charY)
        charY = np.reshape(charY,(-1,))
        return charVocab,id2char,char2id,charX,charY
    
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

    def savechar(self,padlen):
        charVocab,id2char,char2id,charX,charY = self.chardata()
        #print(charVocab,id2char,char2id,charY,charX)
        f = h5py.File("charData"+str(padlen)+".hdf5","w")
        f.create_dataset('X',data=charX)
        f.create_dataset('Y',data=charY)
        d = shelve.open("charData"+str(padlen)+".data")
        d['Vocab'] = charVocab
        d['id2c'] = id2char
        d['c2id'] = char2id
        d.close()
        f.close()
    def saveword(self,padlen):
        wordVocab,id2word,word2id,wordX,wordY = self.worddata()
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
        self.saveword(padlen=padlen)
        print('save word done....')

data = Data('test.txt')
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