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
import matplotlib.pyplot as plt
from decimal import *
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
########################  LAPLACE SMOOTHING ################################
############################################################################

class LaplaceSmoothing():
	def __init__(self, train_corpus, test_corpus):
		self.train_corpus = train_corpus
		self.test_corpus = test_corpus
		self._prepare()

	def	_prepare(self):
		flat_train_corpus = [item for sublist in self.train_corpus for item in sublist]
		self.freq_1gram = nltk.FreqDist(flat_train_corpus)
		number_of_tweets = len(self.train_corpus)
		self.length_corpus = sum(self.freq_1gram.values())-2*number_of_tweets
		self.vocab_corpus = len(set(flat_train_corpus))-2
		unigram_prob = lambda word: self.freq_1gram[word] / self.length_corpus

		self.cfreq_2gram = nltk.ConditionalFreqDist(nltk.bigrams(flat_train_corpus))
		cprob_2gram = nltk.ConditionalProbDist(self.cfreq_2gram, nltk.MLEProbDist)
		bigram_prob = lambda word1, word2: cprob_2gram[word1].prob(word2)

	def unigram_prob_with_add1smoothing(self,word): 
		return Decimal((self.freq_1gram[ word] + 1))/Decimal((self.length_corpus + self.vocab_corpus))

	def bigram_prob_with_add1smoothing(self, word1, word2): 
		return Decimal((1+self.cfreq_2gram[word1][word2]))/Decimal((len(self.cfreq_2gram)+sum(self.cfreq_2gram[word1].values())))

	def perplexity(self,ngram):
		e = Decimal(0.0)
		count = 0
		for sentence in self.test_corpus:
			for i in range(ngram - 1, len(sentence)):
				context = sentence[i - ngram + 1:i]
				token = sentence[i]
				if len(context)==0:
					p=self.unigram_prob_with_add1smoothing(token)
				elif len(context)==1:
					p=self.bigram_prob_with_add1smoothing(context[0], token)
				# e += -p*Decimal(math.log(p , 2))
				e += Decimal(math.log(p))
				count += 1
		entropy = e / Decimal(count - 2*len(test_corpus))
		return pow(Decimal(2.0), entropy)

############################################################################
########################  GOOD TURING SMOOTHING ############################
############################################################################

class GoodTuring():
	def __init__(self, train_corpus, test_corpus):
		self.train_corpus = train_corpus
		self.test_corpus = test_corpus
		self._prepare()

	def	_prepare(self):
		flat_train_corpus = [item for sublist in self.train_corpus for item in sublist]
		self.freq_1gram = nltk.FreqDist(flat_train_corpus)
		self.freq_2gram=nltk.FreqDist(nltk.bigrams(flat_train_corpus))
		self.length_corpus = sum(self.freq_1gram.values())-2*len(self.train_corpus)
		unigram_prob = lambda word: self.freq_1gram[word] / self.length_corpus

		self.cfreq_2gram = nltk.ConditionalFreqDist(nltk.bigrams(flat_train_corpus))
		cprob_2gram = nltk.ConditionalProbDist(self.cfreq_2gram, nltk.MLEProbDist)
		bigram_prob = lambda word1, word2: cprob_2gram[word1].prob(word2)

		self.number_of_unigrams_occuring_n_time = nltk.FreqDist(flat_train_corpus).r_Nr()
		self.number_of_bigrams_occuring_n_time = nltk.FreqDist(nltk.bigrams(flat_train_corpus)).r_Nr()
		self.total_bigrams = 0
		for key, val in self.number_of_bigrams_occuring_n_time.items():
			self.total_bigrams += key*val
	
	def unigram_prob_with_good_turing(self, word):
		unigram_freq = self.freq_1gram[word]
		new_count = Decimal(unigram_freq+1)*Decimal(self.number_of_unigrams_occuring_n_time[unigram_freq+1])/Decimal(self.number_of_unigrams_occuring_n_time[unigram_freq])
		prob = new_count/Decimal(self.length_corpus)
		return prob

	def bigram_prob_with_good_turing(self, word1, word2):
		good_turing=nltk.SimpleGoodTuringProbDist(self.freq_2gram)
		return good_turing.prob((word1, word2))

	def perplexity(self):
		e = Decimal(0)
		count = 0
		for sentence in self.test_corpus:
			for i in range(1, len(sentence)):
				first = sentence[i-2]
				second = sentence[i-1]
				p = self.bigram_prob_with_good_turing(first, second)
				e += -Decimal(p)*Decimal(math.log(Decimal(p) , 2))
				count = count +1
				print count
		entropy = e / Decimal(count)
		return pow(Decimal(2.0), entropy)



############################################################################
#########################  KNESER-NEY SMOOTHING ############################
############################################################################

class KneyserNey():
	def __init__(self, train_corpus, test_corpus):
		self.train_corpus = train_corpus
		self.test_corpus = test_corpus
		self._prepare()

	def	_prepare(self):
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

############################################################################
#################################  MAIN  ###################################
############################################################################
def unigram_prob_with_add1smoothing(word):
	return Decimal((freq_unigram[word] + 1))/Decimal((len(train_corpus) + len(set(train_corpus))))

def bigram_prob_with_add1smoothing(word1, word2): 
	return Decimal((1+cfreq_bigram[word1][word2]))/Decimal((len(cfreq_bigram)+sum(cfreq_bigram[word1].values())))

def trigram_prob_with_add1smoothing(word1, word2, word3): 
	return Decimal((1+cfreq_trigram[(word1,word2)][word3]))/Decimal((len(cfreq_trigram)+sum(cfreq_trigram[(word1,word2)].values())))

def perplexity(ngram):
	e = Decimal(0.0)
	count = 0
	for i in range(ngram - 1, len(test_corpus)):
		context = test_corpus[i-ngram+1:i]
		token = test_corpus[i]
		if len(context)==0:
			p=unigram_prob_with_add1smoothing(token)
		elif len(context)==1:
			p=bigram_prob_with_add1smoothing(context[0], token)
		elif len(context)==2:
			p=trigram_prob_with_add1smoothing(context[0],context[1], token)
		e += Decimal(math.log(p))
		count += 1
	entropy = e / Decimal(count)
	return pow(Decimal(2.0), -entropy)

if __name__ == "__main__":
	loadcorpus = LoadCorpus(filename = 'processed_data/100.txt')
	corpus = loadcorpus.corpus[:1000]
	x =int(90*len(corpus)/100)
	train = corpus[:x]
	test = corpus[x:]

	train_corpus = []
	n_tweets = len(train)
	for i in range(0,n_tweets):
		line = train.pop()
		line_tokens = word_tokenize(line.lower())
		line_tokens = list(pad_sequence(line_tokens, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
		train_corpus.extend(line_tokens)

	test_corpus = []
	n_tweets = len(test)
	for i in range(0,n_tweets):
		line = test.pop()
		line_tokens = word_tokenize(line.lower())
		line_tokens = list(pad_sequence(line_tokens, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
		test_corpus.extend(line_tokens)

	freq_unigram = nltk.FreqDist(nltk.ngrams(flatten(train),1))
	cfreq_bigram = nltk.ConditionalFreqDist(nltk.ngrams(train_corpus,2))
	trigrams = nltk.ngrams(train_corpus,3)
	conditional_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
	cfreq_trigram = nltk.ConditionalFreqDist(conditional_pairs)
	print perplexity(1)
	print perplexity(2)
	print perplexity(3)


# Chap 3

	# import networkx as nx
	# import matplotlib
	# from nltk.corpus import wordnet as wn

	# def traverse(graph, start, node):
	#     graph.depth[node.name] = node.shortest_path_distance(start)
	#     for child in node.hyponyms():
	#         graph.add_edge(node.name, child.name) [1]
	#         traverse(graph, start, child) [2]

	# def hyponym_graph(start):
	#     G = nx.Graph() [3]
	#     G.depth = {}
	#     traverse(G, start, start)
	#     return G

	# def graph_draw(graph):
	#     nx.draw_graphviz(graph,
	#          node_size = [16 * graph.degree(n) for n in graph],
	#          node_color = [graph.depth[n] for n in graph],
	#          with_labels = False)
	#     matplotlib.pyplot.show()

	#     >>> dog = wn.synset('dog.n.01')
	# 	>>> graph = hyponym_graph(dog)
	# 	>>> graph_draw(graph)

# Chap 4







	# laplace =  LaplaceSmoothing(train_corpus, test_corpus)
	# print laplace.perplexity(ngram=2)

	# goodturing = GoodTuring(train_corpus, test_corpus)
	# print goodturing.perplexity()
	
	# kneyserney = KneyserNey(train_corpus, test_corpus)
	# print kneyserney.perplexity()