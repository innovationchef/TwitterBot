def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import re
import math
import cPickle
import itertools
from numpy import c_, exp, log, inf, NaN, sqrt
import numpy as np
import pandas as pd
from collections import OrderedDict
import nltk
from nltk.util import *
from nltk.tokenize import *
from nltk.probability import *
import scipy
from scipy import linalg
from scipy.optimize import curve_fit
from decimal import *
from smoothing.smoothers import *
getcontext().prec = 25

############################################################################
############################  TEXT CLEANING ################################
############################################################################

class LoadCorpus:
    def __init__(self, filename=None, dump_directory = None):
        self.filename = filename
        self.dump_directory = dump_directory
        self.corpus = []
        if (self.filename is not None) and (self.dump_directory is None):
            self.read_sample()
        if (self.dump_directory is not None) and (self.filename is None):
            self.read_dump_data()

    def read_sample(self):
        with open(self.filename, 'rb') as file:
            file_content = file.readlines()
            self.corpus = [line.rstrip('\n') for line in file_content]

    def read_dump_data(self):
        cwd = os.getcwd()
        indir = cwd + '/'+self.dump_directory
        i = 0
        for root, dirs, filenames in os.walk(indir):
            for f in filenames:
                filetext = []
                filename = indir + '/' + f
                with open(filename, 'rb') as file:
                    file_content = file.readlines()
                    self.corpus = [line.rstrip('\n') for line in file_content]

############################################################################
#################################  MAIN  ###################################
############################################################################

if __name__ == "__main__":
    loadcorpus = LoadCorpus(filename = 'processed_data/100.txt')
    corpus = loadcorpus.corpus[:1000]
    x =int(90*len(corpus)/100)
    train = corpus[:x]
    test = corpus[x:]

    smoothed = LaplaceSmoothing(train, test)

    print smoothed.perplexity(1)
    print smoothed.perplexity(2)
    print smoothed.perplexity(3)

    # laplace =  LaplaceSmoothing(train_corpus, test_corpus)
    # print laplace.perplexity(ngram=2)

    # goodturing = GoodTuring(train_corpus, test_corpus)
    # print goodturing.perplexity()

    # kneyserney = KneyserNey(train_corpus, test_corpus)
    # print kneyserney.perplexity()
