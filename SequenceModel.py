import numpy as np
import math

import six
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner


class SequenceModel:

    def __init__(self, GTsActivities, GTs, n_activities, MATCH_SCORE=3, MISSMATCH_SCORE=-2, TOKEN_SCORE=-2):
        self.GTsActivities = GTsActivities
        self.GTs = GTs
        self.n_activities = n_activities

        self.MATCH_SCORE = MATCH_SCORE
        self.MISSMATCH_SCORE = MISSMATCH_SCORE
        self.TOKEN_SCORE = TOKEN_SCORE

    def predict(self, seq, seq_idx=-1):
        scores = self.CompareSeqToGTs(seq, seq_idx)
        # print(scores)
        if seq_idx != -1:
            scores.insert(seq_idx, -math.inf)
        max_score = np.argmax(scores)
        # print(max_score)
        activity_predicted = self.GTsActivities[max_score]
        return activity_predicted


    # compare 2 sequences and get score
    def CompareTwoSequences(self, sequence_a, sequence_b):
        # Create a vocabulary and encode the sequences
        v = Vocabulary()
        aEncoded = v.encodeSequence(sequence_a)
        bEncoded = v.encodeSequence(sequence_b)

        # Create a scoring and align the sequences using global aligner.
        # scoring = SimpleScoring(2, -1)
        # aligner = GlobalSequenceAligner(scoring, -2)  # TODO: understand the theory
        scoring = SimpleScoring(self.MATCH_SCORE, self.MISSMATCH_SCORE)
        aligner = GlobalSequenceAligner(scoring, self.TOKEN_SCORE)
        score = aligner.align(aEncoded, bEncoded)

        return score

    # compare sequence to all GT sequences. Output: list of scores
    def CompareSeqToGTs(self, seq, seq_idx):
        all_compares_scores = []
        for GT_index, GT in enumerate(self.GTs):
            if GT_index != seq_idx:
                all_compares_scores.append(self.CompareTwoSequences(seq, GT))

        return all_compares_scores

