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
    """
    The baseline model.
    """

    def __init__(self, pos_tags, words, training_set):
        """
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        """

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words) + 1
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i, word) in enumerate(words + [RARE_WORD])}

        self.E_prob = np.matrix([])
        self.pi_y = np.matrix([])

    def translate_idx(self, e_count):
        pos_str = re.split("\|", e_count.e)[1]
        pos_idx = self.pos2i[pos_str]
        word_str = re.split("\|", e_count.e)[0]
        try:
            word_idx = self.word2i[word_str]
        except KeyError:
            word_idx = self.word2i[RARE_WORD]
        value = e_count.value
        return ([word_idx, pos_idx, value])

    def predict_pos(self, sentence):
        """
        MAP ALGORITHM
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        """
        tags = []
        for (i, word_str) in enumerate(sentence):
            try:
                word_idx = self.word2i[word_str]  # finds the index for the word of interest
            except KeyError:
                word_idx = self.word2i[RARE_WORD]
            pos_idx = np.multiply(self.pi_y, self.E_prob[word_idx]).argmax()
            tags.append(self.pos_tags[pos_idx])
        return (tags)

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

    DF = pd.DataFrame()
    for row in training_set:
        mat = np.matrix([row[1], row[0]]).T
        df = pd.DataFrame(mat)
        DF = DF.append(df, ignore_index=True)

    e_pairs = DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1)  # combine pos-word
    # pairs have format word_str|pos_str
    e_count = e_pairs.value_counts()
    e_count = e_count.reset_index()
    e_count.columns = ['e', 'value']

    # translate pairs to numeric index-pairs for matrix
    tripel = e_count.apply(lambda x: model.translate_idx(x), axis=1)

    # fill matrix row = word, column = pos
    E_count = np.matrix(np.zeros([len(model.word2i), len(model.pos2i)]))
    for row in tripel:
        E_count[row[0], row[1]] = E_count[row[0], row[1]] + row[2]

    # create E_prob matrix probability P(word, pos)
    E_prob = np.nan_to_num(E_count / E_count.sum(axis=1))

    # P(pos)
    pi_y = E_count.sum(axis=0) / E_count.sum(axis=0).sum()

    return([E_prob, pi_y])

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


score = []
a = 43758
for i in range(a, a+1000):
    sentence = data[i][1]
    tags = [START_STATE]
    tags_est = pd.Series(sentence).apply(lambda x: find_tag(x, tags, pi))

    # to measure accuracy: (how many percent of pos were right)
    score_i = (np.array(tags_est) == np.array(data[i][0])).sum()/len(data[i][0])
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
        self.word2i = {word:i for (i,word) in enumerate(words + [RARE_WORD])}

        self.E_prob = np.matrix([])
        self.pi_y = np.matrix([])
        self.T_prob = np.matrix([])
        self.T_start = np.matrix([])

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''

        # TODO: YOUR CODE HERE

    def translate_e_idx(self, e_count):
        pos_str = re.split("\|", e_count.e)[1]
        pos_idx = self.pos2i[pos_str]
        word_str = re.split("\|", e_count.e)[0]
        try:
            word_idx = self.word2i[word_str]
        except KeyError:
            word_idx = self.word2i[RARE_WORD]
        value = e_count.value
        return ([word_idx, pos_idx, value])

    def translate_t_idx(self, tag_freq):
        pos_str = re.split("\|", tag_freq.tag)[0]
        pos_r_idx = self.pos2i[pos_str]
        pos_str = re.split("\|", tag_freq.tag)[1]
        pos_c_idx = self.pos2i[pos_str]
        value = tag_freq.value
        return ([pos_r_idx, pos_c_idx, value])

    def predict_pos(self, sentence):
        '''
        VITERBI ALGORITHM
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        tags = [START_STATE]

        for (i, word_str) in enumerate(sentence):
            prev_tag = tags[-1]

            if prev_tag == START_STATE:
                T_part = np.log(self.T_start)
                pi     = np.zeros(3)
            else:
                T_part = np.log(self.T_prob[self.pos2i[prev_tag]])

            try:
                pi = pi.max() + np.log(self.E_prob[self.word2i[word_str]]) + T_part
            except KeyError:  # RARE_WORD
                pi = pi.max() + np.log(self.E_prob[self.word2i[RARE_WORD]]) + T_part

            if pi.argmax() == (-1)*np.inf:   #Baseline Model
                print('baseline')
                pi = pi.max() + np.log(self.pi_y) + T_part

            tags.append(pos[pi.argmax()])

        return(tags[1:])


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
    T_DF, E_DF = pd.DataFrame(), pd.DataFrame()
    for row in training_set:
        # transition pairs: (pos_i-1|pos_i)
        t = np.matrix([([START_STATE] + row[0]), (row[0] + [END_STATE])]).T
        t_df = pd.DataFrame(t)
        T_DF = T_DF.append(t_df, ignore_index=True)
        # emission pairs: (word_i|pos_i)
        e = np.matrix([row[1], row[0]]).T
        e_df = pd.DataFrame(e)
        E_DF = E_DF.append(e_df, ignore_index=True)

    e_pairs = E_DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1)  # combine pos-word
    # pairs have format word_str|pos_str
    e_count = e_pairs.value_counts()
    e_count = e_count.reset_index()
    e_count.columns = ['e', 'value']

    # translate pairs to numeric index-pairs for matrix
    tripel = e_count.apply(lambda x: model.translate_e_idx(x), axis=1)

    # fill matrix row = word, column = pos
    E_count = np.matrix(np.zeros([len(model.word2i), len(model.pos2i)]))
    for row in tripel:
        E_count[row[0], row[1]] = E_count[row[0], row[1]] + row[2]

    # create E_prob matrix probability P(word, pos)
    E_prob = np.nan_to_num(E_count / E_count.sum(axis=1))

    # P(pos)
    pi_y = E_count.sum(axis=0) / E_count.sum(axis=0).sum()

    #--------------------------------------------------------------------------------------------------------------
    # same for pos-pairs
    model.pos2i = {pos:i for (i,pos) in enumerate([START_STATE] + model.pos_tags + [END_STATE])}
    tag_pairs = T_DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1)  # combine Tags to pairs

    tag_freq = tag_pairs.value_counts() / len(tag_pairs)
    #tag_freq = pickle.load(open('HMM_tag_freq-90.pkl', 'rb'))
    tag_freq = tag_freq.reset_index()
    tag_freq.columns = ['tag', 'value']

    # create T_freq-matrix row: pos(i), column: pos(i-1)
    tripel = tag_freq.apply(lambda x: model.translate_t_idx(x), axis=1)
    T_prob = np.matrix(np.zeros([len(model.pos2i), len(model.pos2i)]))

    for row in tripel:
        T_prob[row[0], row[1]] = T_prob[row[0], row[1]] + row[2]

    #biring T_freq on 44-pos-states size
    T_start = T_prob[model.pos2i[START_STATE], 1:-1]

    # remove start and end state again
    T_prob = T_prob[1:-1, 1:-1]
    model.pos2i = {pos:i for (i,pos) in enumerate(model.pos_tags)}

    return([E_prob, pi_y, T_start, T_prob])



#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


# find sequence of y that maximizes score
training_data = data[:43757] #90%

#DF = pd.DataFrame()
#for row in training_data:
 #   mat = np.matrix([row[1]]).T
  #  df = pd.DataFrame(mat)
   # DF = DF.append(df, ignore_index=True)

#words_count = DF[0].value_counts()
#words_count.to_pickle('words_count_90.pkl')

words_count = pickle.load(open('words_count_90.pkl', 'rb'))

words_idx_rare = (words_count <= 3)
words_rare = words_count[words_idx_rare].index.tolist()  # 31928
words_used = words_count[~words_idx_rare].index.tolist() # 14969
words_used = words_used + [RARE_WORD]

word2i = {word:i for (i,word) in enumerate(words_used)}
pos2i  = {pos:i for (i,pos) in enumerate(pos)}

def find_tag(tags, word_str, w):
    score = []
    prev_tag = tags[-1]

    try:
        e_offset = len(pos2i)*word2i[word_str]      # e(|pos|*word(i))
    except KeyError:
        e_offset = len(pos2i)*word2i[RARE_WORD]     # e(|pos|*|words|) It's a rare word

    if prev_tag == START_STATE:
        t_offset = len(pos2i)*len(word2i) + len(pos2i)*len(pos2i)       # e(|words|*|pos|) + t(|pos|*|pos|)
    else:
        t_offset = len(pos2i)*len(word2i) + len(pos2i)*pos2i[prev_tag]  # e(|words|*|pos|) + t(|pos|*pos(i-1))

    for i in range(0,len(pos2i)):  # to test which of the pos would give best result
        e_idx = e_offset + i
        t_idx = t_offset + i

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

        # handle RARE_WORDS
        try:
            e_idx = len(pos2i)*word2i[word_str] + pos2i[pos_str]
        except KeyError:
            e_idx = len(pos2i)*word2i[RARE_WORD] + pos2i[pos_str]

        # handle START_STATE because is not included in pos2i
        if i == 0:
            t_offset = len(pos2i)*len(word2i) + len(pos2i)*len(pos2i)
        else:
            prev_tag = tags[i-1]
            t_offset = len(pos2i)*len(word2i) + len(pos2i)*pos2i[prev_tag]
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
    
training_data = data[:43757] #90%

w = train_MEMM(training_data = training_data, eta = 0.2)
# test if makes some sense
w[len(pos2i)*word2i['the']+pos2i['DT']]
w[len(pos2i)*word2i[',']+pos2i[',']]
w[len(pos2i)*word2i['as']+pos2i['IN']]
w[len(pos2i)*word2i['is']+pos2i['VBZ']]
w[len(pos2i)*word2i[RARE_WORD]+pos2i['DT']]


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


def find_frequent_words(training_set, threshold=4):
    """
    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param threshold: included, if threshold=4 it means that a word has to appear at least 4 times to be considered
            in learning
    :return: iterable sequence of words that appeared often enough in training set
    """
    DF = pd.DataFrame()
    for row in training_set:
        mat = np.matrix([row[1]]).T
        df = pd.DataFrame(mat)
        DF = DF.append(df, ignore_index=True)

    words_count = DF[0].value_counts()
    words_used = words_count[(words_count > threshold)].index.tolist()
    return(words_used)


def performance_test(test_set, model):
    """
    determines performance of model based on how many pos-tags were predicted correctly
    :param test_set:
    :param model:
    :return: score: percentage of correct pos-tags
    """
    score = []
    for i in range(0, len(test_set)):
        sentence = test_set[i][1]
        tags = model.predict_pos(sentence)

        # to measure accuracy: (how many percent of pos were right)
        score_i = (np.array(tags) == np.array(test_set[i][0])).sum()/len(tags)
        score.append(score_i)

    final_score = sum(score)/len(score)
    print(final_score)

if __name__ == '__main__':

    data_example()
    
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    # TODO: smarter way of choosing 90% of data
    # according to training set size and test set size

    #training_set = data[0:43757]  # 90% band of data 43757
    training_set = data[:10000]
    test_set = data[43758:]

    # words that were used in training set:
    # words_used = find_frequent_words(training_set)
    # pd.Series(words_used).to_pickle('words_used_90.pickle')
    words_used = pickle.load(open('words_used_90.pickle', 'rb')).tolist()

    # define baseline model
    bl = Baseline(pos_tags=pos, words=words_used, training_set=training_set)

    # training for baseline model, find ML-estimates for transition and emission matrix
    bl.E_prob, bl.pi_y = baseline_mle(training_set, bl)

    score = performance_test(test_set, bl)

    #-------------------------------------------------------------------------------------------------------------------
    hmm = HMM(pos_tags=pos, words=words_used, training_set=training_set)
    hmm.E_prob, hmm.pi_y, hmm.T_start, hmm.T_prob = hmm_mle(training_set, hmm)
    tags = hmm.predict_pos(sentence=sentence)

    score = performance_test(test_set, hmm)
    #TODO: something wrong with hmm - score is 80% but at baseline is 87%

