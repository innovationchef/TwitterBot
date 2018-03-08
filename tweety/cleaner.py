import re
import itertools

class TextCleaning:
	def __init__(self):
		self.corpus = []

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
		speclchars = re.compile(r'[_~@\^\*\(\)\+={}\|\[\]\\/]')
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
								 (3:-\)) | (3:\))  #devil""", flags = re.VERBOSE|re.IGNORECASE)

		if nonenglish.search(s) is None:
			newline_removed = re.sub(newline, ' ', s)
			replace_sign = re.sub(r'&gt;', '>', newline_removed)
			replace_sign = re.sub(r'&lt;', '<', replace_sign)
			weblinks_removed = re.sub(weblinks, '', replace_sign)
			mentions_removed = re.sub(mentions, ' <person> ', weblinks_removed)
			# hashtags_removed = re.sub(hashtags, ' <hashtag> ', mentions_removed)
			repeats_removed = ''.join(''.join(s)[:2] for _, s in itertools.groupby(mentions_removed))
			brackets_removed = re.sub(brackets, '', repeats_removed)
			emoticons_removed = re.sub(emoticons, '', brackets_removed)
			numbers_removed = re.sub(numbers, ' <number> ', emoticons_removed)
			speclchars_removed = re.sub(speclchars, ' ', numbers_removed)
			tabs_removed = re.sub(tabs, ' ', speclchars_removed)
			spaces_removed = re.sub(spaces, ' ', tabs_removed)
			apostrophe_removed = self.apostrophe(spaces_removed)
			repeatedpunctuations_removed = self.repeated_punctuations(apostrophe_removed)
			unspacedpunctuations_removed = self.unspacedpunctuations(repeatedpunctuations_removed)
			unspacedhyphens_removed =  self.unspacedhyphens(unspacedpunctuations_removed)	
			if len(unspacedhyphens_removed.lstrip())>0:
				final_text = unspacedhyphens_removed.lstrip()
				if final_text.lower()[:3] == "rt ": 
					return final_text.lower()[3:]
				else:
					return final_text.lower()
