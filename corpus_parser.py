from nltk import word_tokenize as tokenize
import sklearn
import random
import numpy as np

# return a list of tokenized sentences
def tokenize_file(filename):
    sents = []
    file = open(filename,'r')
    for line in file:
        sents.append(tokenize(line.lower()))    
    return sents

# puts words of each sentence into one list
def unpack_sents(sents):
    tokens = []
    for sentence in sents:
        tokens.extend(sentence)
    return tokens

# combines the raw tokens of all sentence lists
def all_words(corpora):
    tokens = []
    for corpus in corpora:
        tokens.extend(unpack_sents(corpus))
    return tokens

# convert a sequence of tokens into a binary numpy vector
def vectorize(sentence, vocab):
    vector = np.zeros([len(vocab), 1])
    for i in range(len(vocab)):
        # if word is present in sentence
        if vocab[i] in sentence:
            # mark a 1 in the vector
            vector[i] = 1
    return np.transpose(vector)

def parse_files():
    weather_sents = tokenize_file('./corpus/commands/weather.txt')
    twitter_sents = tokenize_file('./corpus/commands/trending.txt')
    news_sents = tokenize_file('./corpus/commands/headlines.txt')
    gibberish_sents = tokenize_file('./corpus/commands/gibberish.txt')

    corpora = [weather_sents, twitter_sents, news_sents, gibberish_sents]
    words = all_words(corpora)
    vocab = list(set(words))

    global dataset
    dataset = []
    for i in range(len(corpora)):
        v = np.zeros([len(corpora), 1])
        v[i] = 1
        for sentence in corpora[i]:
            dataset.append([vectorize(sentence, vocab),
                            np.transpose(v)])

    return len(vocab), len(corpora), len(datset)

# partition datset into training and test sets
def make_sets():
    # make the test set of one input patter
    global test_set
    i = random.randrange(len(dataset))
    test_set = copy.deepcopy(dataset[i])
    # make the training set of all other patterns
    global train_set
    train_set = []

    for j in range(len(dataset)):
        if i is not j:
            train_set
    

# return the next pattern to feed forward
def next_stimulus():
    i = random.randrange(len(train_set))
    return train_set[i]
