import tensorflow as tf
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
import h5py
import shelve
import tensorflow as tf
import numpy as np
from p7_TextCNN_model import TextCNN
import os
#import word2vec
import pickle
import codecs

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",5,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 200, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1, "Rate of decay for learning rate.") #0.65一次衰减多少
#tf.app.flags.DEFINE_integer("num_sampled",50,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",100,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每1轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 256, "number of filters") #256--->512
tf.app.flags.DEFINE_boolean("sentence_len",50,"the length of the sentence.")
filter_sizes=[1,2,3,4,5,6,7] #[1,2,3,4,5,6,7]
def main(_):
    out = codecs.open('out1.csv','w','utf8')
    f = h5py.File('../datautils/charTest50.hdf5','r')
    X = f['X'].value
    d = shelve.open('../datautils/charData50.data')
    Vocab,vocabulary_index2word,vocabulary_word2index= d['Vocab'],d['id2c'],d['c2id']
    d = shelve.open('../datautils/idTest50')
    idx = d['id']
    vocab_size = len(vocabulary_word2index)+1
    with tf.Session() as sess:
        #Instantiate Model
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        saver.restore(sess,'checkpoint/model.ckpt-17')
        for index,i in enumerate(X):
            pred = sess.run([textCNN.predictions],feed_dict={textCNN.input_x:[i],textCNN.dropout_keep_prob: 1})
            out.write(str(idx[index])+','+str(pred[0][0])+'\n')

if __name__ == "__main__":
    tf.app.run()