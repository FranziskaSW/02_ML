import pickle
import pandas as pd
import numpy as np
import re
import random
from scipy.misc import logsumexp

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


#########################################################################################
#                                   Baseline Model                                      #
#########################################################################################

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

        self.E_prob = np.matrix([])  # for emission matrix (|words|*|pos|)
        self.pi_y = np.matrix([])    # for probability of pos
                                     # both are updated through training (baseline_mle)

    def translate_idx(self, e_count):
        """
        Translates the word-pos-string pairs into indices to fill E_prob matrix
        :param e_count: string type index of word-pos pair and count
        :return: list of indeces and count-value to fill E_prob matrix
        """
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
        Given an iterable sequence of word sequence, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequence (sentence).
        :return: iterable sequence of PoS tag sequences.
        """
        tags = []
        for word_str in sentence:
            try:
                word_idx = self.word2i[word_str]  # finds the index for the word of interest
            except KeyError:                      # if word is not in dictionary
                word_idx = self.word2i[RARE_WORD] # handle it as RARE_WORD
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
    for row in training_set:                  # for every entry of the training data
        mat = np.matrix([row[1], row[0]]).T   # combine the word with the pos-tag in matrix
        df = pd.DataFrame(mat)                #
        DF = DF.append(df, ignore_index=True) # create pd.DataFrame to use apply-functions

    e_pairs = DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1)  # combine pos-word
                                              # output is a pd.Series of index-string-pairs
                                              # pairs have format word_str|pos_str
    e_count = e_pairs.value_counts()          # count how often every pair appears
    e_count = e_count.reset_index()           #
    e_count.columns = ['e', 'value']          #

    # translate pairs to numeric index-pairs for matrix
    tripel = e_count.apply(lambda x: model.translate_idx(x), axis=1)

    # fill matrix row = word, column = pos
    E_count = np.matrix(np.zeros([len(model.word2i), len(model.pos2i)]))
    for row in tripel:
        E_count[row[0], row[1]] = E_count[row[0], row[1]] + row[2]

    # create E_prob matrix probability P(word, pos)
    E_prob = np.nan_to_num(E_count / E_count.sum(axis=1))

    # P(pos): Probability of pos
    pi_y = E_count.sum(axis=0) / E_count.sum(axis=0).sum()

    # update model
    model.E_prob = E_prob
    model.pi_y = pi_y

#########################################################################################
#                            Hidden Markov Model (HMM)                                  #
#########################################################################################
        
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

        self.E_prob = np.matrix([])    # for emission matrix (|words|*|pos|)
        self.pi_y = np.matrix([])      # for probability of pos
        self.T_prob = np.matrix([])    # for transition matrix (|pos|*|pos|)
        self.T_start = np.matrix([])   # for transition of START_STATE to first word
        self.T_end = np.matrix([])     # for transition of last word to END_STATE
                                       # all are updated through training (baseline_mle)

    def sample(self, n):
        '''
        Sample sequence of n words from the HMM.
        :param n: desired length of word sequence
        :return: A list of word sequence.
        '''
        tags = []
        words = []
        word_str = 'his'

        a = np.log(self.T_start) + np.log(self.E_prob[self.word2i[word_str]])
        pos_idx = a.argmax().item(0)          # which pos leads to maximum value
        pos_str= self.pos_tags[pos_idx]       # translate index into string of pos
        tags.append(pos_str)
        words.append(word_str)

        for i in range(1,n):
            a = np.log(self.T_prob[pos_idx,:]) + np.log(self.E_prob) # previous tag known, want to estimate current tag
            max_values = a.max(axis=0)        # maximum value for every pos
            argmax_word = a.argmax(axis=0)    # which word gave this maximum (for every pos)
            pos_idx = max_values.argmax(axis=1).item(0)  # position of maximum values of possible pos
            word_idx = argmax_word.item(pos_idx)         # which word belongs to this pos
            word_str = self.words[word_idx]
            pos_str = self.pos_tags[pos_idx]
            words.append(word_str)
            tags.append(pos_str)
        return(words)


    def translate_e_idx(self, e_count):
        """
        Translates the word-pos-string pairs into indices to fill E_prob matrix
        :param e_count: string type index of word-pos pair and count
        :return: list of indeces and count-value to fill E_prob matrix
        """
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
        """
        Translates the pos-pos-string pairs into indices to fill T_prob matrix
        :param tag_freq: string type index of pos-pos pair and count
        :return: list of indeces and count-value to fill T_prob matrix
        """
        pos_str = re.split("\|", tag_freq.tag)[0]
        pos_r_idx = self.pos2i[pos_str]
        pos_str = re.split("\|", tag_freq.tag)[1]
        pos_c_idx = self.pos2i[pos_str]
        value = tag_freq.value
        return ([pos_r_idx, pos_c_idx, value])

    def predict_pos(self, sentence):
        '''
        VITERBI ALGORITHM
        Given an iterable sequence of word sequence, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequence (sentence).
        :return: iterable sequence of PoS tag sequence.
        '''

        word_str = sentence[0]

        try:
            self.word2i[word_str]
        except KeyError:                     # if word is not found in dictionary
            word_str = RARE_WORD             # handle it as RARE_WORD

        # pi for first transition (Start_tag to first word)
        pi = (np.log(self.T_start) + np.log(self.E_prob[self.word2i[word_str]])).T

        pos_idx_list = []
        pos_idx = pi.argmax(axis=0).item(0)   # pos that has highest probability
        pos_idx_list.append(pos_idx)

        # pi for the other transitions
        for i in range(1,len(sentence)):
            word_str = sentence[i]
            try:
                self.word2i[word_str]
            except KeyError:
                word_str = RARE_WORD

            a  = pi + np.log(self.T_prob) + np.log(self.E_prob[self.word2i[word_str]])
            pi = a.max(axis=0).T
            pos_idx = pi.argmax(axis=0).item(0)

            if pi.max() == (-1)*np.inf:
                # something is wrong, it happens quite often that all the terms cancel out
                # to not get only zero-vectors following this case, I  go back to the baseline model
                pi = np.log(self.pi_y) + np.log(self.E_prob[self.word2i[word_str]])
                pos_idx = pi.argmax()

            pos_idx_list.append(pos_idx)

        # translate pos indices into pos tags
        tags = []
        for i in argmax_list:
            tag_i = pos[i]
            tags.append(tag_i)

        return(tags)

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
    for row in training_set:                  # for every entry of the training_set
        # transition pairs: (pos_i-1|pos_i)   # combine pos tag of previous word with pos tag of current word
        t = np.matrix([([START_STATE] + row[0]), (row[0] + [END_STATE])]).T
        t_df = pd.DataFrame(t)                #
        T_DF = T_DF.append(t_df, ignore_index=True)
        # emission pairs: (word_i|pos_i)      # also combine word with its pos tag
        e = np.matrix([row[1], row[0]]).T     #
        e_df = pd.DataFrame(e)                #
        E_DF = E_DF.append(e_df, ignore_index=True)

    e_pairs = E_DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1)  # combine pos-word
                                              # output is a pd.Series of index-string-pairs
                                              # pairs have format word_str|pos_str
    e_count = e_pairs.value_counts()          #
    e_count = e_count.reset_index()           #
    e_count.columns = ['e', 'value']          #

    # translate pairs to numeric index-pairs for matrix
    tripel = e_count.apply(lambda x: model.translate_e_idx(x), axis=1)

    # fill matrix row = word, column = pos
    E_count = np.matrix(np.zeros([len(model.word2i), len(model.pos2i)]))
    for row in tripel:
        E_count[row[0], row[1]] = E_count[row[0], row[1]] + row[2]

    # create E_prob matrix probability P(word, pos)
    E_prob = np.nan_to_num(E_count / E_count.sum(axis=1))

    # P(pos): Probability of pos
    pi_y = E_count.sum(axis=0) / E_count.sum(axis=0).sum()

    # do the same for pos-pairs
    # expand model pos2i-dictionary because here we also consider START_STATE and END_STATE
    model.pos2i = {pos:i for (i,pos) in enumerate([START_STATE] + model.pos_tags + [END_STATE])}
    tag_pairs = T_DF.apply(lambda x: str(x[0]) + '|' + str(x[1]), axis=1)  # combine tags to pairs

    tag_freq = tag_pairs.value_counts() / len(tag_pairs)
    tag_freq = tag_freq.reset_index()
    tag_freq.columns = ['tag', 'value']

    # create T_freq-matrix row: pos(i), column: pos(i-1)
    tripel = tag_freq.apply(lambda x: model.translate_t_idx(x), axis=1)
    T_prob = np.matrix(np.zeros([len(model.pos2i), len(model.pos2i)]))

    for row in tripel:
        T_prob[row[0], row[1]] = T_prob[row[0], row[1]] + row[2]

    # bring T_freq on 44-pos-states size, drop the rows for START_STATE and END_STATE
    T_start = T_prob[model.pos2i[START_STATE], 1:-1]
    T_end   = T_prob[1:-1, model.pos2i[END_STATE]]
    T_prob = T_prob[1:-1, 1:-1]

    # update model
    model.pos2i = {pos:i for (i,pos) in enumerate(model.pos_tags)}
    model.E_prob = E_prob
    model.T_prob = T_prob
    model.T_start = T_start
    model.T_end = T_end
    model.pi_y = pi_y

#########################################################################################
#                       Maximum Entropy Markov Model (MEMM)                             #
#########################################################################################

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
        :param w: weights for prediction in form of dictionary
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words + [RARE_WORD])}
        self.phi = phi
        self.w = {}


    def predict_pos(self, sentence):
        '''
        VITERBI ALGORITHM
        Given an iterable sequence of word sequence, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequence (sentence).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequence.
        '''
        len_pos, len_words = len(self.pos2i), len(self.word2i)
        t_scores = []

        #calculate part for y_0 --> y_1 (since y_0 is known as START_STATE)

        word_str = sentence[0]
        score = []

        # find the indices for transition and emission state
        try:
            e_offset = len_pos * self.word2i[word_str]   # e(|pos|*word(i))
        except KeyError:                                 # if word not in dictionary
            e_offset = len_pos * self.word2i[RARE_WORD]  # e(|pos|*|words|) handle as RARE_WORD

        t_offset = len_pos * len_words + len_pos * len_pos

        for i in range(0, len_pos):  # iterate over pos-tags of word_str (pos of current word)
            e_idx = e_offset + i
            t_idx = t_offset + i

            # check if weight-dictionary already has an entry at this position
            try:
                score_i = self.w[e_idx]
            except KeyError:
                score_i = 0
            try:
                score_i = score_i + self.w[t_idx]
            except KeyError:
                score_i = score_i
            score.append(score_i)         # iterated over pos of current state,
                                          # given the START_STATE as previous
        score = score - logsumexp(score)  # normalize score
        score = np.matrix(score)          #
        t_scores.append(score)            #

        # now calculate transitions/emissions for different current pos tags (i) and previous pos tags(j)
        for word_idx in range(1, len(sentence)):
            word_str = sentence[word_idx]
            scores_matrix = np.zeros([len_pos, len_pos])

            for j in range(0,len_pos):    # iterate over previous tags (j)
                score = []                #
                try:                      # check if word is in dictionary
                    e_offset = len_pos * self.word2i[word_str]   # e(|pos|*word(i))
                except KeyError:          # if not handle as rare word
                    e_offset = len_pos * self.word2i[RARE_WORD]  # e(|pos|*|words|)

                t_offset = len_pos * len_words * self.pos2i[self.pos_tags[j]]

                for i in range(0, len_pos):  # to test which of the current pos tags (i) would give best result
                    e_idx = e_offset + i
                    t_idx = t_offset + i

                    # check if weight-dictionary already has an entry at this position
                    try:
                        score_i = self.w[e_idx]
                    except KeyError:
                        score_i = 0
                    try:
                        score_i = score_i + self.w[t_idx]
                    except KeyError:
                        score_i = score_i
                    score.append(score_i)        # iterated over pos of current state

                score = score - logsumexp(score) # normalize score
                scores_matrix[j] = score         # iterated over pos of previous state
            t_scores.append(scores_matrix)

        # now find the maximum probabilities of the transitions and find the argmax (index of pos)
        pi = t_scores[0].T                       # for the first state
        pos_idx = pi.argmax()                    #
        pos_idx_list = []                        #
        pos_idx_list.append(pos_idx)             #

        for i in range(1, len(t_scores)):        # now do the same for the following transitions
            t_i = t_scores[i]                    #
            a = pi + t_i                         #
            pi = a.max(axis=1)                   #
            pos_idx = a.argmax(axis=1).T         #
                                                 #
            df = pd.DataFrame(pos_idx.T)         # to handle the case if different pos lead to an maximum
            val_count = df[0].value_counts()     # count the occurance
            value = val_count.index[0]           # take the one that appeared the most often
            pos_idx_list.append(value)           #

        # translate pos indices into pos tags
        tags = []
        for i in pos_idx_list:
            tag_i = self.pos_tags[i]
            tags.append(tag_i)

        return(tags)


def phi(sentence, tags, model):
    """
    uses the words and tags to calculate a mapping of the index of the phi vector to its value
    where only relevant indices are mapped to their value
    :param sentence: iterable sequence of words of the sentence
    :param tags: iterable sequence of tags
    :param model: the memm model
    :return: phi as dictionary
    """
    phi_dict = {}
    word2i = model.word2i
    pos2i = model.pos2i

    for i in range(0, len(sentence)):   # for every word in the sentence
        word_str = sentence[i]          # get word
        pos_str = tags[i]               # get pos

        # find emission index
        try:
            e_idx = len(pos2i) * word2i[word_str] + pos2i[pos_str]
        except KeyError:                # if word is not in dictionary, handle it as RARE_WORD
            e_idx = len(pos2i) * word2i[RARE_WORD] + pos2i[pos_str]

        # find the offset of the transition
        # handle START_STATE because is not included in pos2i
        if i == 0:
            t_offset = len(pos2i) * len(word2i) + len(pos2i) * len(pos2i)
        else:
            prev_tag = tags[i - 1]
            t_offset = len(pos2i) * len(word2i) + len(pos2i) * pos2i[prev_tag]
        t_idx = t_offset + pos2i[pos_str]        # combine transition offset with pos to get transition index

        phi_i = {e_idx: 1, t_idx: 1}    # this is the phi-vector for the given word-pos-pair

        for key in phi_i.keys():        # add this new entry to the phi-dictionary
            try:
                phi_dict[key] = phi_dict[key] + phi_i[key]
            except KeyError:
                phi_dict[key] = phi_i[key]

    return (phi_dict)

def perceptron(training_set, model, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM. (with update)
    """
    for i in range(0, len(training_set)):
        sentence = training_set[i][1]
        tags_known = training_set[i][0]

        # most likely sequence of pos given the sentence
        tags_est = model.predict_pos(sentence)

        # calculate the phi-vectors for known tags and estimated tags
        phi_known = phi(sentence=sentence, tags=tags_known, model=model)
        phi_est = phi(sentence=sentence, tags=tags_est, model=model)

        phi_diff = phi_known        # calculate difference between the two phi-vectors
        for key in phi_est.keys():  #
            try:                    #
                phi_diff[key] = phi_known[key] - phi_est[key]
            except KeyError:        #
                phi_diff[key] = -phi_est[key]

        # update model
        for key in phi_diff.keys():
            try:
                model.w[key] = model.w[key] + eta * phi_diff[key]
            except KeyError:
                model.w[key] = eta * phi_diff[key]

def find_frequent_words(training_set, threshold=4):
    """
    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param threshold: included, if threshold=4 it means that a word has to appear at least 4 times to be considered
            in learning
    :return: iterable sequence of words that appeared often enough in training set
    """
    DF = pd.DataFrame()
    for row in training_set:         # for every sentence in training_set
        mat = np.matrix([row[1]]).T  # write list of words into matrix
        df = pd.DataFrame(mat)       # and transform this matrix into pd.DataFrame
        DF = DF.append(df, ignore_index=True)

    words_count = DF[0].value_counts() # count the values in this DataFrame
    words_used = words_count[(words_count > threshold)].index.tolist() # only keep the ones that appeared more often
                                                                       # than threshold-times (including)
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

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    # shuffle data since we don't know how it was generated
    # random.shuffle(data)
    # pd.Series(data).to_pickle('data.pickle')
    data = pickle.load(open('data.pickle', 'rb'))
    data = data.tolist()

    #training_set = data[0:43757]  # 90% of data
    training_set = data[0:24309]   # 50% of data
    training_set = data[0:12155]   # 25% of data
    test_set = data[43758:]        # last 10% of data
    # words that were used in training set:
    words_used = find_frequent_words(training_set)
    pd.Series(words_used).to_pickle('words_used.pickle')
    words_used = pickle.load(open('words_used.pickle', 'rb')).tolist()

    #-------------------------------------------------------------------------------------------------------------------
    # define baseline model
    bl = Baseline(pos_tags=pos, words=words_used, training_set=training_set)

    # training for baseline model, find ML-estimates for transition and emission matrix
    bl.E_prob, bl.pi_y = baseline_mle(training_set, bl)
    pd.DataFrame(bl.E_prob).to_pickle('bl_E_prob_50.pickle')
    pd.DataFrame(bl.pi_y).to_pickle('bl_pi_y_50.pickle')

    score = performance_test(test_set, bl) # 0.8663122861529187 (50% of data as training_set)
                                           # 0.8316422798311665 (25% of data as training_set)
    print(score)

    #-------------------------------------------------------------------------------------------------------------------
    hmm = HMM(pos_tags=pos, words=words_used, training_set=training_set)
    hmm.E_prob, hmm.pi_y, hmm.T_start, hmm.T_end, hmm.T_prob = hmm_mle(training_set, hmm)
    pd.DataFrame(hmm.E_prob).to_pickle('hmm_E_prob_50.pickle')
    pd.DataFrame(hmm.T_start).to_pickle('hmm_T_start_50.pickle')
    pd.DataFrame(hmm.T_end).to_pickle('hmm_T_end_50.pickle')
    pd.DataFrame(hmm.T_prob).to_pickle('hmm_T_prob_50.pickle')
    pd.DataFrame(hmm.pi_y).to_pickle('hmm_pi_y_50.pickle')

    score = performance_test(test_set, hmm) # 0.8989316537845912 (50% of data as training_set)
                                            # 0.8798304154335393 (25% of data as training_set)
    print(score)

    #-------------------------------------------------------------------------------------------------------------------
    memm = MEMM(pos_tags=pos, words=words_used, training_set=training_set, phi = 1)

    # train model and save parameters (w)
    perceptron(training_set, memm)
    with open('memm_w.pickle', 'wb') as f:
        pickle.dump(memm.w, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('memm_w.pickle', 'rb') as f:
        memm.w = pickle.load(f)

    memm.w[len(memm.pos2i) * memm.word2i['the'] + memm.pos2i['DT']]
    memm.w[len(memm.pos2i) * memm.word2i[','] + memm.pos2i[',']]
    memm.w[len(memm.pos2i) * memm.word2i['as'] + memm.pos2i['IN']]
    memm.w[len(memm.pos2i) * memm.word2i['is'] + memm.pos2i['VBZ']]
    memm.w[len(memm.pos2i) * memm.word2i[RARE_WORD] + memm.pos2i['DT']]
    memm.w[len(memm.pos2i) * memm.word2i[RARE_WORD] + memm.pos2i['NN']]

    score = performance_test(test_set, memm) # 0.8235056389297039 (25% of data as training_set)
    print(score)

