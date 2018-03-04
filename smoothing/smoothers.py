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
getcontext().prec = 25

############################################################################
############################  LAPLACE SMOOTHING ############################
############################################################################

class LaplaceSmoothing():
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def _pad_and_tokenize(self, corpus, n_tweets, ngram):
        tokens = []
        for i in range(0,n_tweets):
            line = corpus.pop()
            line_tokens = word_tokenize(line.lower())
            line_tokens = list(pad_sequence(line_tokens, ngram, pad_left=True, \
                pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            tokens.extend(line_tokens)
        return tokens

    def _prepare(self, ngram):
        self.train_corpus = self._pad_and_tokenize(self.train[:], len(self.train), ngram)
        self.test_corpus = self._pad_and_tokenize(self.test[:], len(self.test), ngram)

        if ngram == 1:
            self.freq_unigram = nltk.FreqDist(nltk.ngrams(self.train_corpus,1))
        elif ngram == 2: 
            self.cfreq_bigram = nltk.ConditionalFreqDist(nltk.ngrams(self.train_corpus,2))
        elif ngram == 3:
            trigrams = nltk.ngrams(self.train_corpus,3)
            conditional_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
            self.cfreq_trigram = nltk.ConditionalFreqDist(conditional_pairs)
        else:
            print("Error")

    def unigram_prob_with_add1smoothing(self, word):
        return Decimal((self.freq_unigram[word] + 1))/Decimal((len(self.train_corpus) \
            + len(set(self.train_corpus))))

    def bigram_prob_with_add1smoothing(self, word1, word2):
        return Decimal((1+self.cfreq_bigram[word1][word2]))/Decimal((len(self.cfreq_bigram) \
            +sum(self.cfreq_bigram[word1].values())))

    def trigram_prob_with_add1smoothing(self, word1, word2, word3):
        return Decimal((1+self.cfreq_trigram[(word1,word2)][word3]))/Decimal( \
            (len(self.cfreq_trigram)+sum(self.cfreq_trigram[(word1,word2)].values())))

    def perplexity(self, ngram):
        self._prepare(ngram)
        e = Decimal(0.0)
        count = 0
        for i in range(ngram - 1, len(self.test_corpus)):
            context = self.test_corpus[i-ngram+1:i]
            token = self.test_corpus[i]
            if len(context)==0:
                p=self.unigram_prob_with_add1smoothing(token)
            elif len(context)==1:
                p=self.bigram_prob_with_add1smoothing(context[0], token)
            elif len(context)==2:
                p=self.trigram_prob_with_add1smoothing(context[0],context[1], token)
            e += Decimal(math.log(p))
            count += 1
        entropy = e / Decimal(count)
        return pow(Decimal(2.0), -entropy)

############################################################################
########################  GOOD TURING SMOOTHING ############################
############################################################################

class GoodTuring():
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def _pad_and_tokenize(self, corpus, n_tweets, ngram):
        tokens = []
        for i in range(0,n_tweets):
            line = corpus.pop()
            line_tokens = word_tokenize(line.lower())
            line_tokens = list(pad_sequence(line_tokens, ngram, pad_left=True, \
                pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            tokens.extend(line_tokens)
        return tokens

    def _prepare(self, ngram):
        self.train_corpus = self._pad_and_tokenize(self.train[:], len(train), ngram)
        self.test_corpus = self._pad_and_tokenize(self.test[:], len(test), ngram)

        if ngram == 1:
            self.freq_unigram = nltk.FreqDist(nltk.ngrams(self.train_corpus,1))
        elif ngram == 2: 
            self.cfreq_bigram = nltk.FreqDist(nltk.ngrams(self.train_corpus,2))
        elif ngram == 3:
            trigrams = nltk.ngrams(self.train_corpus,3)
            conditional_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
            self.cfreq_trigram = nltk.FreqDist(conditional_pairs)
        else:
            print("Error")

    def unigram_prob_with_good_turing(self, word):
        good_turing=nltk.SimpleGoodTuringProbDist(self.freq_unigram)
        return good_turing.prob(word)

    def bigram_prob_with_good_turing(self, word1, word2):
        good_turing=nltk.SimpleGoodTuringProbDist(self.cfreq_bigram)
        return good_turing.prob((word1, word2))

    def trigram_prob_with_good_turing(self, word1, word2, word3):
        good_turing=nltk.SimpleGoodTuringProbDist(self.cfreq_trigram)
        return good_turing.prob(((word1, word2), word3))

    def perplexity(self, ngram):
        self._prepare(ngram)
        e = Decimal(0.0)
        count = 0
        for i in range(ngram - 1, len(self.test_corpus)):
            context = self.test_corpus[i-ngram+1:i]
            token = self.test_corpus[i]
            if len(context)==0:
                p=self.unigram_prob_with_good_turing(token)
            elif len(context)==1:
                p=self.bigram_prob_with_good_turing(context[0], token)
            elif len(context)==2:
                p=self.trigram_prob_with_good_turing(context[0],context[1], token)
            e += Decimal(math.log(p))
            count += 1
        entropy = e / Decimal(count)
        return pow(Decimal(2.0), -entropy)

############################################################################
#########################  KNESER-NEY SMOOTHING ############################
############################################################################

class KneyserNey():
    def __init__(self, train_corpus, test_corpus):
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self._prepare()

    def    _prepare(self):
        self.freq_of_coming_first = {}
        self.freq_of_coming_second = {}
        flat_train_corpus = [item for sublist in self.train_corpus for item in sublist]
        self.freq_1gram = nltk.FreqDist(flat_train_corpus)
        self.length = sum(self.freq_1gram.values())-2*len(self.train_corpus)
        self.cfreq_2gram = nltk.ConditionalFreqDist(nltk.bigrams(flat_train_corpus))
        bigrams = list(nltk.bigrams(flat_train_corpus))
        for a,b in bigrams:
            if a in self.freq_of_coming_first:
                self.freq_of_coming_first[a]+=1
            else:
                self.freq_of_coming_first[a]=1
            if b in self.freq_of_coming_second:
                self.freq_of_coming_second[b]+=1
            else:
                self.freq_of_coming_second[b]=1

    def KNsmoothing(self, bigram_freq,context_freq,word_freq,freq_of_coming_first,freq_of_coming_second,word_bigram_types,N):
        if context_freq>0:
            return (Decimal(max(Decimal(0.0),Decimal(bigram_freq) - Decimal(0.5)))/Decimal(context_freq)) + \
            (Decimal(0.5)/Decimal(context_freq))*(freq_of_coming_first*freq_of_coming_second)/Decimal(word_bigram_types)
        else:
            return Decimal(word_freq)/Decimal(N)

    def perplexity(self):
        word_bigram_types = sum(self.freq_of_coming_second.values())
        N = self.length - self.freq_1gram['<s>']
        e = Decimal(0)
        count = 0
        for sentence in test_corpus:
            for i in range(1, len(sentence)):
                bigCount = self.cfreq_2gram[sentence[i-1]][sentence[i]]
                smoothed = 0
                if bigCount==0:
                    # Out of vocabulary case. Skip
                    if self.freq_1gram[sentence[i]]==0:
                        continue
                    # If the context is unseen
                    elif (self.freq_1gram[sentence[i-1]] == 0) and (self.freq_1gram[sentence[i]] > 0):
                        smoothed = Decimal(self.freq_1gram[sentence[i]])/Decimal(N)
                    # If the bigram is unseen
                    elif (self.freq_1gram[sentence[i-1]] > 0) and (self.freq_1gram[sentence[i]] > 0):
                        smoothed = self.KNsmoothing(bigCount,self.freq_1gram[sentence[i-1]],self.freq_1gram[sentence[i]],self.freq_of_coming_first[sentence[i-1]],self.freq_of_coming_second[sentence[i]],word_bigram_types,N)
                else:
                    # When the bigram has count > 0
                    smoothed = self.KNsmoothing(bigCount,self.freq_1gram[sentence[i-1]],self.freq_1gram[sentence[i]],self.freq_of_coming_first[sentence[i-1]],self.freq_of_coming_second[sentence[i]],word_bigram_types,N)
                # e += -Decimal(smoothed)*Decimal(math.log(Decimal(smoothed) , 2))
                e += Decimal(math.log(smoothed))
                count = count +1
        entropy = e / Decimal(count)
        return pow(Decimal(2.0), entropy)
