import pickle
import pandas as pd
import numpy as np
import datetime
import re
from itertools import compress
import os

cwd = os.getcwd()

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'


def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])

    print(pos)


class Baseline(object):
    '''
    The baseline model.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}

        # TODO: YOUR CODE HERE

    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''

        # TODO: YOUR CODE HERE

############################################################################
#     Data Preprocessing
############################################################################
training_set = data[0:43757]  # 90% band of data 43757

DF = pd.DataFrame()
for row in training_set:
    mat = np.matrix([row[1], row[0]]).T
    df = pd.DataFrame(mat)
    DF = DF.append(df, ignore_index=True)

e_pairs = DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1) # combine Tags to pairs

e_count = e_pairs.value_counts()
e_count = e_count.reset_index()
e_count.columns = ['e', 'value']

# index has format word_str|pos_str
# now fill matrix: row=word - column = pos


def translate_idx(e_freq):
    pos_str   = re.split("\|", e_freq.e)[1]
    pos_idx   = pos.index(pos_str)
    word_str  = re.split("\|", e_freq.e)[0]
    word_idx  = words.index(word_str)
    value = e_freq.value
    return([word_idx, pos_idx, value])

    
# create e-count matrix
    
tripel = e_count.apply(lambda x: translate_idx(x), axis=1)
E_count = np.matrix(np.zeros([len(words),len(pos)]))
for row in tripel:
    print(row)
    E_count[row[0], row[1]] = row[2]

#pd.DataFrame(E_count).to_pickle('baseline_E-count_20.pkl')

E_count = pickle.load(open('baseline_E-count_90.pkl', 'rb'))
E_count = np.matrix(E_count)

words_count    = E_count.sum(axis=1) # for infrequent words
words_idx_rare = (words_count <= 0)
rare_idx_list  = words_idx_rare.flatten().tolist()[0]
words_rare     = list(compress(words, rare_idx_list))
used_idx_list  = (~words_idx_rare).flatten().tolist()[0]
words_used     = list(compress(words, used_idx_list))

#reduce E_count to used words - remove rare words
E_count_used = E_count[used_idx_list, :]

#create E-prob matrix
E_prob = np.nan_to_num(E_count_used / words_count[used_idx_list])

pi_y        = E_count_used.sum(axis=0)/words_count[used_idx_list].sum()



def Likelihood(word_str):
    try:
        idx = words_used.index(word_str)  # finds the index for the word of interest
        pos_idx = np.multiply(pi_y, E_prob[idx]).argmax()
        pos_str = pos[pos_idx]

    except (IndexError ,ValueError):
        pos_str = RARE_WORD
    return(pos_str)

score = []
a = 43758
for i in range(a, a+100):
    tags = []
    for word_str in data[i][1]:
        tag = Likelihood(word_str)
        tags.append(tag)

    # to measure accuracy: (how many percent of pos were right)
    score_i = (np.array(tags) == np.array(data[i][0])).sum()/len(data[i][0])
    score.append(score_i)

final_score = sum(score)/len(score)
print(final_score)

def baseline_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. You may implement
            the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """

    # TODO: YOUR CODE HERE

'''
DF = pd.DataFrame()
for i in range(0,43757):
    t = data[i][0]
    mat = np.matrix([([START_STATE] + t), (t + [END_STATE])]).T
    df = pd.DataFrame(mat)
    DF = DF.append(df, ignore_index=True)

tag_pairs = DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1) # combine Tags to pairs

tag_freq = tag_pairs.value_counts() / len(tag_pairs)

pd.DataFrame(tag_freq).to_pickle('HMM_tag_freq-90.pkl')

print('next step')

DF = pd.DataFrame()
for i in range(0,9723):
    t = data[i][0]
    mat = np.matrix([([START_STATE] + t), (t + [END_STATE])]).T
    df = pd.DataFrame(mat)
    DF = DF.append(df, ignore_index=True)

tag_pairs = DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1) # combine Tags to pairs

tag_freq = tag_pairs.value_counts() / len(tag_pairs)

pd.DataFrame(tag_freq).to_pickle('HMM_tag_freq-20.pkl')
'''

pos_h = [START_STATE] + pos + [END_STATE]

tag_freq = pickle.load(open('HMM_tag_freq-90.pkl', 'rb'))

tag_freq = tag_freq.reset_index()
tag_freq.columns = ['tag', 'value']

def translate_idx(tag_freq):
    pos_r_str = re.split("\|", tag_freq.tag)[0]
    pos_r     = pos_h.index(pos_r_str)
    pos_c_str = re.split("\|", tag_freq.tag)[1]
    pos_c  = pos_h.index(pos_c_str)
    value = tag_freq.value
    return([pos_r, pos_c, value])

# create T_freq-matrix row: pos(i), column: pos(i-1)

tripel = tag_freq.apply(lambda x: translate_idx(x), axis=1)
T_freq = np.matrix(np.zeros([len(pos_h),len(pos_h)]))

for row in tripel:
    T_freq[row[0], row[1]] = row[2]

T_start = T_freq[0,1:-1]

# remove start and end state again
T_freq = T_freq[1:-1,1:-1]

# E_prob-matrix
# Same as from baseline model but add colum for START_STATUS and END_STATUS
E_count = pickle.load(open('baseline_E-count_90.pkl', 'rb'))
E_count = np.matrix(E_count)

words_count    = E_count.sum(axis=1) #for infrequent words
words_idx_rare = (words_count <= 4)
rare_idx_list  = words_idx_rare.flatten().tolist()[0]
words_rare     = list(compress(words, rare_idx_list))
used_idx_list  = (~words_idx_rare).flatten().tolist()[0]
words_used     = list(compress(words, used_idx_list))

#reduce E_count to used words - remove rare words
E_count_used = E_count[used_idx_list, :]

#create E-prob matrix
E_prob = np.nan_to_num(E_count_used / words_count[used_idx_list])
#Add zero rows for START_ and END_STATE
#E_prob = np.vstack((np.zeros(len(words_used)), E_prob.T, np.zeros(len(words_used)))).T
#create E-prob matrix

pi_y        = E_count_used.sum(axis=0)/words_count[used_idx_list].sum()


def find_tag(word_str, tags, pi):
    prev_tag = tags[-1]
    try:
        if prev_tag == RARE_WORD:
            pi = pi.max() + np.log(E_prob[words_used.index(word_str)])
        else:
            pi = pi.max() + np.log(T_freq[pos.index(prev_tag),:]) + np.log(E_prob[words_used.index(word_str)])
            
        if pi.max() == (-1)*np.inf: # baseline model
            pi = np.log(pi_y) + np.log(E_prob[words_used.index(word_str)])
            
        tag = pos[pi.argmax()]
    except ValueError:
        pi = pi
        tag = RARE_WORD
    return(tag)

def all_tags(sentence):
    tags = []
    word_str = sentence[0]
    try:
        pi = np.log(T_start) + np.log(E_prob[words_used.index(word_str)])
        tags.append(pos[pi.argmax()])
    except ValueError:
        pi = np.log(T_start) + np.log(pi_y)
        tags.append(RARE_WORD)
        
    tags = pd.Series(sentence).apply(lambda x: find_tag(x, tags, pi))
    return(tags)

sentence = data[43830][1] #43830
all_tags(sentence)

    
score = []
a = 43758
for i in range(a, a+300):
    sentence = data[i][1]
    tags = all_tags(sentence)

    # to measure accuracy: (how many percent of pos were right)
    score_i = (np.array(tags) == np.array(data[i][0])).sum()/len(data[i][0])
    score.append(score_i)

final_score = sum(score)/len(score)
print(final_score)




        
class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}

        # TODO: YOUR CODE HERE

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''

        # TODO: YOUR CODE HERE


    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''

        # TODO: YOUR CODE HERE


def hmm_mle(training_set, model):
    
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """

    # TODO: YOUR CODE HERE

# TODO: in baseline and HMM: RARE WORDS before calculation of estimators e,t
# TODO: in baseline and HMM: use word2i and pos2i instead of pos.index(pos_str)

# find sequence of y that maximizes score

training_data = data[:2000]

DF = pd.DataFrame()
for row in training_data:
    mat = np.matrix([row[1]]).T
    df = pd.DataFrame(mat)
    DF = DF.append(df, ignore_index=True)

words_count = DF[0].value_counts()
words_idx_rare = (words_count <= 3)
words_rare = words_count[words_idx_rare].index.tolist()  # 31928
words_used = words_count[~words_idx_rare].index.tolist() # 14969


word2i = {word:i for (i,word) in enumerate(words_used)}
pos2i  = {pos:i for (i,pos) in enumerate(pos)}


def find_tag(tags, word_str, w):
    # TODO: RARE_WORD
    score = []
    prev_tag = tags[-1]

    try:
        e_offset = len(pos2i) * word2i[word_str] # e(|pos|*word(i))
    except KeyError:
        e_offset = len(pos2i)*len(word2i)        # e(|pos|*|words|) It's a rare word

    if prev_tag == START_STATE:
        t_offset = len(pos2i)*len(word2i) + len(pos2i) + len(pos2i)*len(pos2i)       # e(|words|*|pos|) + e_rare(|pos|) + t(|pos|*|pos|)
    else:
        t_offset = len(pos2i)*len(word2i) + len(pos2i) + len(pos2i)*pos2i[prev_tag]  # e(|words|*|pos|) + e_rare(|pos|) + t(|pos|*pos(i-1))

    for i in range(0,len(pos2i)):  # to test which of the pos would give best result
        e_idx = e_offset + i
        t_idx = t_offset + i

        print(e_idx, t_idx)

        try: 
            score_i = w[e_idx]
        except KeyError:
            score_i = 0
        try:
            score_i = score_i + w[t_idx]
        except KeyError:
            score_i = score_i
        score.append(score_i)
    tag = pos[np.array(score).argmax()]
    return(tag)


def phi(sentence, tags):
    
    phi_dict = {}
    for i in range(0,len(tags)):
        word_str = sentence[i]
        pos_str  = tags[i]

        try:
            e_idx = len(pos2i) * word2i[word_str] + pos2i[pos_str]
        except KeyError:
            e_idx = len(pos2i) * len(word2i) + pos2i[pos_str]

        if i == 0:
            t_offset = len(pos2i)*len(word2i) + len(pos2i) + len(pos2i)*len(pos2i)
        else:
            prev_tag = tags[i-1]
            t_offset = len(pos2i)*len(word2i) + len(pos2i) + len(pos2i)*pos2i[prev_tag]
        t_idx = t_offset + pos2i[pos_str]

        phi_i = {e_idx:1, t_idx:1}
        
        for key in phi_i.keys():
            try:
                phi_dict[key] = phi_dict[key] + phi_i[key]
            except KeyError:
                phi_dict[key] = phi_i[key]
        
    return(phi_dict)

    
def train_MEMM(training_data, eta):
    
    w = {}
    for i in range(0,len(training_data)):
        sentence   = training_data[i][1]
        tags_known = training_data[i][0]
        
        # most likely sequence of pos given the sentence
        tags = [START_STATE]
        for word_str in sentence:
            tag = find_tag(tags, word_str, w)
            tags.append(tag)
        tags_est = tags[1:]
        
        
        phi_known = phi(sentence = sentence, tags = tags_known)
        phi_est   = phi(sentence = sentence, tags = tags_est)
        
        phi_diff = phi_known
        for key in phi_est.keys():
            try:
                phi_diff[key] = phi_known[key] - phi_est[key]
            except KeyError:
                phi_diff[key] = -phi_est[key]
        
        for key in phi_diff.keys():
            try:
                w[key] = w[key] + eta*phi_diff[key]
            except KeyError:
                w[key] = eta*phi_diff[key]
    return(w)
    
training_data = data[:1] #43757] #90%

w = train_MEMM(training_data = training_data, eta = 0.2)
# test if makes some sense
w[len(pos2i)*word2i['the']+pos2i['DT']]
w[len(pos2i)*word2i[',']+pos2i[',']]
w[len(pos2i)*word2i['as']+pos2i['IN']]
w[len(pos2i)*word2i[',']+pos2i['DT']]


score = []
a = 43758
for i in range(a, a+4861):
    sentence = data[i][1]
    
    tags = [START_STATE]
    tags = pd.Series(sentence).apply(lambda x: find_tag(tags,x,w))

    # to measure accuracy: (how many percent of pos were right)
    score_i = (np.array(tags) == np.array(data[i][0])).sum()/len(data[i][0])
    score.append(score_i)

final_score = sum(score)/len(score)
print(final_score)


class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, training_set, phi):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        self.phi = phi

        # TODO: YOUR CODE HERE
        
        # something like numbers


    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''

        # TODO: YOUR CODE HERE


def perceptron(training_set, initial_model, w0, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """

    # TODO: YOUR CODE HERE


if __name__ == '__main__':

    data_example()
    
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)
    # TODO: YOUR CODE HERE