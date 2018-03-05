from decimal import *
import smoothing.smoothers
import random
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
        cwd = smoothing.smoothers.os.getcwd()
        indir = cwd + '/'+self.dump_directory
        i = 0
        for root, dirs, filenames in smoothing.smoothers.os.walk(indir):
            for f in filenames:
                filetext = []
                filename = indir + '/' + f
                with open(filename, 'rb') as file:
                    file_content = file.readlines()
                    self.corpus = [line.rstrip('\n') for line in file_content]

############################################################################
#################################  MAIN  ###################################
############################################################################

def _pad_and_tokenize(corpus, n_tweets, ngram):
    tokens = []
    for i in range(0,n_tweets):
        line = corpus.pop()
        line_tokens = smoothing.smoothers.word_tokenize(line.lower())
        line_tokens = list(smoothing.smoothers.pad_sequence(line_tokens, ngram, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        tokens.extend(line_tokens)
    return tokens

def sort_by_value(mydict):
    return sorted(mydict.items(),key = lambda x:x[1],reverse = True)

def counts_cumul(l):
    if len(l)==1:
        return l
    else:
        list1 = []
        for i in range(len(l)):
          if i==0:
            list1.append((l[i][0],l[i][1]))
          else:
            list1.append((l[i][0],l[i][1]+list1[i-1][1]))
        return list1

def next_word(word):
    relevant_words = sort_by_value(word)
    list2 = counts_cumul(relevant_words)
    randnumber = random.randint(1,list2[-1][1])
    for i in range(len(list2)):
        if (list2[i][1]>=randnumber):
                return list2[i][0]

def generate_random_tweet(cfreq, word, steps=7):
    tweet = []
    for i in range(steps):
        if cfreq[word]:
            given_word = cfreq[word]
            tweet.append(next_word(given_word))
        else:
            print "*****************"
    return ' '.join(tweet)

def writing_CFG_grammer(tweets):
    tag_dict = smoothing.smoothers.defaultdict(list)
    tagged_sent = smoothing.smoothers.nltk.pos_tag(tweets.split())
    for word, tag in tagged_sent:
        if tag not in tag_dict:
            tag_dict[tag].append(word)
        elif word not in tag_dict.get(tag):
            tag_dict[tag].append(word)
    for tag, words in tag_dict.items():
        print tag, "->",
        first_word = True
        for word in words:
            if first_word:
                print "\"" + word + "\"",
                first_word = False
            else:
                print "| \"" + word + "\"",
        print ''

if __name__ == "__main__":
    loadcorpus = LoadCorpus(filename = 'processed_data/100.txt')
    corpus = loadcorpus.corpus[:10000]
    x =int(90*len(corpus)/100)
    train = corpus[:x]
    test = corpus[x:]

    ngram = 2

    train_corpus = _pad_and_tokenize(train, len(train), ngram)
    test_corpus = _pad_and_tokenize(test, len(test), ngram)

    cfreq_bigram = smoothing.smoothers.nltk.ConditionalFreqDist(smoothing.smoothers.nltk.ngrams(train_corpus, 2))


    tweets = []
    for i in range(25):
        tweet = generate_random_tweet(cfreq_bigram, 'the')
        tweets.append(tweet)

    writing_CFG_grammer(' '.join(tweets))
    
    grammar = """
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP
    VP -> V NP | VP PP
    Det -> 'DT'
    N -> 'NN'
    V -> 'VBZ'
    P -> 'PP'
    VBG -> "shooting" | "freaking" | "beginning" | "soothing" | "living" 
    RB -> "very" | "smell" | "only" | "axe" | "right" 
    NN -> "computer" | "</s>" | "union" | "person" | "gift" | "head" | "number" | "service" | "toilet" | "swift" | "consumerist" | "basement" | "goal" | "state" | "booth" | "studio" | "office" | "pier" | "night" | "favor" | "music" | "end" | "bandwagon" | "hotel" | "man" | "plane" | "moment" | "song" | "ncaa" | "morning" | "dude" | "cap" | "sun" | "value" | "quanti" | "history" | "future" | "world" | "truth" | "yarn" | "gr" | "bedroom" | "apocalypse" | "eye" | "mafia" | "air" | "door" | "inconvenience" | "way" | "game" | "car" | "party" | "ball" | "shit" | "corner" | "team" | "time" | "locker" | "colour" | "event" | "int" | "wash" | "movie" | "fambo" | "forum" | "stuff" | "half" | "jungle" 
    VBD -> "toast" | "revised" 
    RBS -> "most" 
    JJS -> "cutest" | "latest" | "best" | "strongest" | "greatest" | "newest" 
    VB -> "snowfall" 
    CD -> "one" 
    VBP -> "bus" | "nambur" | "follow" | "quanti" | "salt" | "am" | "play" 
    JJ -> "fambo" | "console" | "stupid" | "i" | "fish" | "perfect" | "funny" | "decimal" | "crown" | "upper" | "red" | "top" | "same" | "lady" | "nambur" | "bitter" | "egyptian" | "new" | "whole" | "snow" | "main" | "queen" | "wash" | "elliptical" | "north" | "sound" | "green" | "next" | "garage" | "hunterian" | "girls" | "national" | "few" | "ball" | "merm" 
    IN -> "abc" | "boss" | "over" 
    VBZ -> "articles" 
    DT -> "the" 
    NNS -> "quarterfinals" | "snacks" | "bengals" | "rationalizations" | "pacers" | "birds" | "stars" | "stairs" | "games" | "bears" | "people" | "girls" | "flinstones" | "words" 
    JJR -> "healthier"
    """
    # simple_grammar = smoothing.smoothers.nltk.CFG.fromstring(grammar)
    # tweet = smoothing.smoothers.nltk.word_tokenize(tweet)
    # POS_tagged_tweet = smoothing.smoothers.nltk.pos_tag(tweet)
    # pos_tags = [pos for (token,pos) in POS_tagged_tweet]
    # parser = smoothing.smoothers.nltk.ChartParser(simple_grammar)
    # tree = parser.parse(pos_tags)
    # print tree
