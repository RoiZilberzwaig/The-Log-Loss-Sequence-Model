#
# import six
# from alignment.sequence import Sequence
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
#
# # Create sequences to be aligned.
# a = Sequence(['background', 'take', 'take', 'take', 'background', 'background', 'take', 'background', 'open', 'open', 'open', 'background', 'background', 'background', 'take', 'take', 'background', 'scoop', 'scoop', 'scoop', 'scoop', 'scoop', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'take', 'background', 'background', 'close', 'close', 'close', 'close', 'close', 'close', 'put', 'put', 'background', 'take', 'take', 'background', 'background', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'background', 'close'])
# b = Sequence(['background', 'take', 'take', 'background', 'background', 'take', 'take', 'put', 'background', 'take', 'open', 'open', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'pour', 'close', 'close', 'take', 'take', 'open', 'open', 'background', 'pour', 'pour', 'close', 'close', 'put', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background', 'background'])
#
# # Create a vocabulary and encode the sequences.
# v = Vocabulary()
# aEncoded = v.encodeSequence(a)
# bEncoded = v.encodeSequence(b)
#
# # Create a scoring and align the sequences using global aligner.
# scoring = SimpleScoring(2, -1)
# aligner = GlobalSequenceAligner(scoring, -2)
# score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
#
# # Iterate over optimal alignments and print them.
# for encoded in encodeds:
#     alignment = v.decodeSequenceAlignment(encoded)
#     # print alignment
#     # print 'Alignment score:', alignment.score
#     # print 'Percent identity:', alignment.percentIdentity()
#     # print
#
#


from copy import deepcopy
import numpy as np
import six

from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
import math

class LogSequenceModel:

    def __init__(self, GTsActivities, GTs, n_activities,  MATCH_SCORE=10, MISSMATCH_SCORE=-1, TOKEN_SCORE=-5):
        self.GTsActivities = GTsActivities
        self.GTs = GTs
        self.n_activities = n_activities

        self.MATCH_SCORE = MATCH_SCORE
        self.MISSMATCH_SCORE = MISSMATCH_SCORE
        self.TOKEN_SCORE = TOKEN_SCORE

    # def predict(self, sf, GTs):
    #     scores = []
    #     for GT in GTs:
    #         scores.append(self.CompareSFToGTi(sf, GT))
    #     min_score = np.argmin(scores)
    #     activity_predicted = self.GTsActivities[min_score]
    #     return activity_predicted

    def predict(self, sf, sf_index=-1):
        log_losses = self.CompareSeqToGTs(sf, sf_index)
        # print(log_losses)
        if sf_index != -1:
            log_losses.insert(sf_index, math.inf)
        min_loss = np.argmin(log_losses)   # TODO: check - argmax gives the right prediction for video 0 for some reason..
        # print(min_loss)
        activity_predicted = self.GTsActivities[min_loss]
        return activity_predicted

    # compare 2 sequences and get score
    def SetSameLength(self, sequence_a, sequence_b):
        # Create a vocabulary and encode the sequences
        v = Vocabulary()
        aEncoded = v.encodeSequence(sequence_a)
        bEncoded = v.encodeSequence(sequence_b)

        # Create a scoring and align the sequences using global aligner.
        # scoring = SimpleScoring(2, 0)  # TODO: consider change penalty - was -1
        # aligner = GlobalSequenceAligner(scoring, -2)  # TODO: understand the theory, gap was -2
        scoring = SimpleScoring(self.MATCH_SCORE, self.MISSMATCH_SCORE)  # TODO: consider change penalty - was -1
        aligner = GlobalSequenceAligner(scoring, self.TOKEN_SCORE)  # TODO: understand the theory, gap was -2
        _, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True) # FeatureA: send index of background (list(v).index(7))

        # Iterate over optimal alignments and print them.
        for encoded in encodeds:
            alignment = v.decodeSequenceAlignment(encoded)

        return alignment.first, alignment.second

    # compare sequence to all GT sequences. Output: list of scores
    def CompareSeqToGTs(self, sf, sf_index):
        all_compares_scores = []
        for idx, GTi in enumerate(self.GTs):
            if sf_index != idx:
                all_compares_scores.append(self.CompareSFToGTi(sf, GTi))

        return all_compares_scores

    # Compare matrix of softmax to one GT sequence. The cost is calculated by -ln(). Output: list of scores.
    def CompareSFToGTi(self, sf_matrix, GT):
        SF_k, GTi_k = self.PrepareSFandGTi(sf_matrix, GT)
        log_score = self.LogCalcSFToGTi(SF_k, GTi_k)
        return log_score


    def PrepareSFandGTi(self, sf_matrix, GT):
        sf_argmax = argmax_of_X(sf_matrix)
        sf_argmax_k, GTi_k = self.SetSameLength(Sequence(sf_argmax), Sequence(GT))

        # take sf_argmax -> SF_k ('-' will get 1s
        SF_k = deepcopy(sf_matrix)
        # Sanity Check
        if len(GTi_k) < max(len(sf_matrix[0]), len(GT)):
            raise Exception('the new len is shorter')
        counter = 0
        for frame in range(len(sf_argmax_k)):
            if sf_argmax_k[frame] == '-':
                [SF_k[i].insert(frame, '-') for i in range(len(SF_k))]
                counter += 1

        return SF_k, GTi_k

    def LogCalcSFToGTi(self,SF_k, GTi_k):
        logLoss = 0
        numOfSpaces = 0
        # trace = [SF_k[GTi_k[i]] for i in range(len(GTi_k))]   # tried to create trace but get '-' in GTi_k (try evalute to understand the problem..)
        for idx in range(len(GTi_k)):
            if SF_k[0][idx] != '-' and GTi_k[idx] != '-':
                logLoss += -np.log(SF_k[GTi_k[idx]][idx])
            # else:
        #         numOfSpaces += 1
        #         logLoss += 10
        #     # else there is no penalty - TODO: consider some penalty..
        # logLoss += -(numOfSpaces*np.log(0.2))
        # # logLoss /= (len(GTi_k) - numOfSpaces)
        return logLoss

def argmax_of_X(X):
    return [int(np.matrix(X).T.argmax(1)[i]) for i in range(len(np.matrix(X).T))]
