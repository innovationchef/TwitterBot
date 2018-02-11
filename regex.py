def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn
import json
import gzip
import re
import os
import math
from decimal import *
from collections import OrderedDict
import itertools

############################################################################
############################  TEXT CLEANING ################################
############################################################################

class TextCleaning:
	def __init__(self, filename=None, main_directory=None, dump_directory = None):
		self.filename = filename
		self.main_directory = main_directory
		self.dump_directory = dump_directory
		self.corpus = []
		if (self.filename is not None) and (self.main_directory is None) and (self.dump_directory is None):
			self.clean_sample()
		if (self.main_directory is not None) and (self.dump_directory is not None):
			self.clean_and_dump_directory()
		if (self.dump_directory is not None) and (self.main_directory is None):
			self.read_dump_data()

	def apostrophe(self, phrase):
		# specific
		phrase = re.sub(re.compile(r"won\'t", flags=re.IGNORECASE), "will not", phrase)
		phrase = re.sub(re.compile(r"can\'t", flags=re.IGNORECASE), "can not", phrase)
		phrase = re.sub(re.compile(r"ain\'t", flags=re.IGNORECASE), "are not", phrase)
		phrase = re.sub(re.compile(r"\'cause", flags=re.IGNORECASE), "because", phrase)
		# general
		phrase = re.sub(re.compile(r"\'re", flags=re.IGNORECASE), " are", phrase)
		phrase = re.sub(re.compile(r"\'s", flags=re.IGNORECASE), " is", phrase)
		phrase = re.sub(re.compile(r"\'d", flags=re.IGNORECASE), " would", phrase)
		phrase = re.sub(re.compile(r"\'ll", flags=re.IGNORECASE), " will", phrase)
		phrase = re.sub(re.compile(r"\'t", flags=re.IGNORECASE), " not", phrase)
		phrase = re.sub(re.compile(r"\'ve", flags=re.IGNORECASE), " have", phrase)
		phrase = re.sub(re.compile(r"\'m", flags=re.IGNORECASE), " am", phrase)
		phrase = re.sub(re.compile(r"\'em", flags=re.IGNORECASE), " them", phrase)
		phrase = re.sub(re.compile(r"n\'t", flags=re.IGNORECASE), " not", phrase)
		phrase = re.sub(re.compile(r"ma\'am", flags=re.IGNORECASE), "madam", phrase)
		phrase = re.sub(re.compile(r"o\'clock", flags=re.IGNORECASE), " of the clock", phrase)
		phrase = re.sub(re.compile(r"\'t\'ve", flags=re.IGNORECASE), " not have", phrase)
		return phrase

	def repeated_punctuations(self, phrase):
		repeatedpunctuations = re.compile(r'[\?\.,:!;$%&_-]{2,}')
		if re.search(repeatedpunctuations, phrase):
			identify_repeatedpunctuations = re.search(repeatedpunctuations, phrase)
			phrase = phrase[:identify_repeatedpunctuations.start()+1]+phrase[identify_repeatedpunctuations.end():]
			phrase = self.repeated_punctuations(phrase)
		return phrase

	def unspacedpunctuations(self, phrase):
		unspacedpunctuations_begin = re.compile(r'([\.\?,:!;\'\"&]\w)')
		unspacedpunctuations_end = re.compile(r'(\w[\.\?,:!;\'\"&])')
		if re.search(unspacedpunctuations_begin, phrase):
			identify_unspacedpunctuations_begin = re.search(unspacedpunctuations_begin, phrase)
			phrase = phrase[:identify_unspacedpunctuations_begin.start()+1]+' '+phrase[identify_unspacedpunctuations_begin.start()+1:]
			phrase = self.unspacedpunctuations(phrase)
		if re.search(unspacedpunctuations_end, phrase):
			identify_unspacedpunctuations_end = re.search(unspacedpunctuations_end, phrase)
			phrase = phrase[:identify_unspacedpunctuations_end.start()+1]+' '+phrase[identify_unspacedpunctuations_end.start()+1:]
			phrase = self.unspacedpunctuations(phrase)
		return phrase

	def unspacedhyphens(self, phrase):
		unspacedhyphens_begin = re.compile(r'(\W-\w)')
		unspacedhyphens_end = re.compile(r'(\w-\W)|(\w-$)')
		if re.search(unspacedhyphens_begin, phrase):
			identify_unspacedhyphens_begin = re.search(unspacedhyphens_begin, phrase)
			phrase = phrase[:identify_unspacedhyphens_begin.start()+2]+' '+phrase[identify_unspacedhyphens_begin.start()+2:]
			phrase = self.unspacedhyphens(phrase)
		if re.search(unspacedhyphens_end, phrase):
			identify_unspacedhyphens_end = re.search(unspacedhyphens_end, phrase)
			phrase = phrase[:identify_unspacedhyphens_end.start()+1]+' '+phrase[identify_unspacedhyphens_end.start()+1:]
			phrase = self.unspacedhyphens(phrase)
		return phrase

	def preprocess(self, s):
		nonenglish = re.compile(r'[^\x00-\x7F]')
		newline = re.compile(r'[\n]')
		tabs = re.compile(r'\t')
		spaces = re.compile(r'(\s{2,})')
		weblinks = re.compile(r"""(https?://?(www\.)?(\w+).+) | #http wale links
									((www\.)[.^\S]+) | #www wale link
									([.^\S]+(\.com)[.^\S]+)|([.^\S]+(\.co\.)[.^\S]+)|([.^\S]+)(\.com) #.com wale links
								""", flags = re.IGNORECASE | re.VERBOSE)
		mentions = re.compile(r'@[.\S]+', flags = re.IGNORECASE)
		hashtags = re.compile(r'#[.\S]+', flags = re.IGNORECASE)
		brackets = re.compile(r'[\[\(\{](.*?)[\]\)\}]', flags = re.IGNORECASE)
		numbers = re.compile(r'(\W+)?([0-9]+)(\W+)?')
		speclchars = re.compile(r'[_~@#\^\*\(\)\+={}\|\[\]<>\\/]')
		emoticons = re.compile(r"""(:\)) | (=\]) | (=\)) | (:\]) | (-:\)) | #smile
								 (:\() | (=\[) | (=\() | (:\[) | (-:\() | #sad
								 (:p) | (:-p) | (=p) | #sticking out
								 (=D) | (:D) | (xD) | (:-D) | #Grinning
								 (:-O) | (:O) | #gasping
								 (;\)) | #winking
								 (_/\\_) | (\\m/) | #respect
								 (>:-\() | (>:\() #grumpy
								 (:-/) | (:/) | #unsure
								 (:\'\() | (:\'\)) | #crying
								 (\^_\^) | (\^-\^) | (\*_\*) | (\*-\*) | #kiki
								 (-_-) | (-__-) | (-\.-) | (:\|) | (>_>) | (<_<) | #squinting
								 (o\.O) | (O\.o) | (0\.0) | (O_o) | (u_u) | #confused
								 (>-:O) | (>:O) | #upset
								 (8\)) | (B-\)) | (B\)) | (8-\|) | (8\|) | (B-\|) | (B\|) | #wearing
								 (O:-\)) | (O:\)) | #angel
								 (<3) | #heart
								 (:3) | #disgust
								 (3:-\)) | (3:\)) |  #devil
								 """, flags = re.VERBOSE|re.IGNORECASE)

		if nonenglish.search(s) is None:
			newline_removed = re.sub(newline, ' ', s)
			replace_sign = re.sub(r'&gt;', '>', newline_removed)
			replace_sign = re.sub(r'&lt;', '<', replace_sign)
			weblinks_removed = re.sub(weblinks, '', replace_sign)
			mentions_removed = re.sub(mentions, '', weblinks_removed)
			hashtags_removed = re.sub(hashtags, '', mentions_removed)
			repeats_removed = ''.join(''.join(s)[:2] for _, s in itertools.groupby(hashtags_removed))
			emoticons_removed = re.sub(emoticons, '', repeats_removed)
			brackets_removed = re.sub(brackets, '', emoticons_removed)
			numbers_removed = re.sub(numbers, ' NAMBUR ', brackets_removed)
			speclchars_removed = re.sub(speclchars, ' ', numbers_removed)
			tabs_removed = re.sub(tabs, ' ', speclchars_removed)
			spaces_removed = re.sub(spaces, ' ', tabs_removed)
			apostrophe_removed = self.apostrophe(spaces_removed)
			repeatedpunctuations_removed = self.repeated_punctuations(apostrophe_removed)
			unspacedpunctuations_removed = self.unspacedpunctuations(repeatedpunctuations_removed)
			unspacedhyphens_removed =  self.unspacedhyphens(unspacedpunctuations_removed)	
			if len(unspacedhyphens_removed.lstrip())>0:
				final_text = unspacedhyphens_removed.lstrip()
				return final_text

	def clean_sample(self):
		with gzip.open(self.filename, 'rb') as file:
			file_content = file.readlines()
			lines = [line.rstrip('\n') for line in file_content]

		for line in lines:
			j = json.loads(line)['text'].encode('utf-8')
			# print j
			if self.preprocess(j) is None: pass
			else:	self.corpus.append(self.preprocess(j))

	def clean_and_dump_directory(self):
		cwd = os.getcwd()
		indir = cwd + '/'+self.main_directory
		i = 0
		for root, dirs, filenames in os.walk(indir):
			for f in filenames:
				filetext = []
				filename = indir + '/' + f
				with gzip.open(filename, 'rb') as file:																																																																																																																																																																																																																																																																																																																
					file_content = file.readlines()
					lines = [line.rstrip('\n') for line in file_content]

				for line in lines:
					j = json.loads(line)['text'].encode('utf-8')
					if self.preprocess(j) is None: pass
					else:	self.corpus.append(self.preprocess(j))
				print i, "  ===>  ", f
				i+=1																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																
				if i%100 == 0: 
					f = open(self.dump_directory+"/" + str(i) + ".txt", "w")
					for line in self.corpus:
						 f.write("%s\n" % line)
					f.close()
					del self.corpus
					self.corpus = []
		f = open(self.dump_directory + '/' + str(i) + ".pkl", "w")
		f.write(cPickle.dumps(self.corpus))
		f.close()
		del self.corpus

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

if __name__ == "__main__":
	TextCleaning(main_directory = 'json.gold', dump_directory = 'processed_data')
	