import ast
import numpy as np
import pandas as pd

import DataGTEA
from DataGTEA import Data_Drinks, DataSandwiches

import SequenceModel
import LogSequenceModel
import NoiseReducerModel

import six
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner


def txt_to_matrix(text):
    text = ((",".join(text.split(' '))).replace(',,', ', ').replace(',', ', '))
    matrix = ast.literal_eval(text)
    return np.array(matrix)


def label_preprocess(label, n_activities = None):
    if n_activities is None:
        return [item if len(item) ==1 else [item[0]] if len(item) > 1 else [-1] for item in label]
    return [item if len(item) ==1 else [item[0]] if len(item) > 1 else [n_activities - 1] for item in label]


# n_activities are total activities+1 for background which is the last row in the matrix
# activities are 0-6 and background is 7 so in total 8
def label_to_matrix(label, n_activities = 8):
    width = len(label)
    label_mat = [[0] * width for _ in range(n_activities)]
    
    for i in range(width):
        label_mat[label[i][0]][i] = 1
        
    return label_mat 


def swap_first_and_last_elements(probability_matrix):
    new_mat = []
    for timestamp in probability_matrix:
        rotated_timestamp = np.roll(np.array(timestamp), -1).tolist()
        new_mat.append(rotated_timestamp)
    return new_mat


def transpose_matix(m):
    return list(map(list, zip(*m)))


def prepare_labels(labels_lst, n_activities = 8):
    matrix_labels = []
    
    for label in labels_lst:
        clean_label = label_preprocess(label)
        label_matrix = label_to_matrix(clean_label, n_activities)
        matrix_labels.append(label_matrix)
    string_labels = convert_matrices_to_string_vectors(matrix_labels, n_activities)    
    return string_labels


def prepare_label_numeric(label, n_activities=None):
    label = label_preprocess(label, n_activities)
    label = [item[0] for item in label]
    label = np.array(label)
    return label


def modify_label_dimensions(label, required_dimension):
    n_additional_entries = required_dimension - label.size
    if n_additional_entries > 0:
        additional_entries = np.ones(n_additional_entries)*7
        label = np.append(label, additional_entries)
    return label

def prepare_labels_numeric(labels, required_dimension, n_activities=None):
    prepared_labels = []
    for label in labels:
        label = prepare_label_numeric(label, n_activities)
        label = modify_label_dimensions(label, required_dimension)
        prepared_labels.append(label)
    return prepared_labels


def prepare_predictions(predictions_lst):
    post_processed_predictions = []
    
    for prediction in predictions_lst:
        pred_matrix = txt_to_matrix(prediction)
        pred_matrix = swap_first_and_last_elements(pred_matrix)
        pred_matrix_transpose = transpose_matix(pred_matrix)
        post_processed_predictions.append(pred_matrix_transpose)
    
    return post_processed_predictions


def accuracy(vec1, vec2, vec_length=-1):
    if vec_length == -1:
        vec_length = vec1.size
    if vec1.size != vec2.size:
        raise ValueError(f'The vectors must have same size! in this case first vector has a size of {vec1.size} while the second is of size {vec2.size}')
    return sum(vec1[0:vec_length] == vec2[0:vec_length]) / vec_length
    # return sum(vec1 == vec2) / vec1.size



def matrix_to_string_vector(trace_matrix, n_activities=8):
    trace = []
    for col in list(zip(*trace_matrix)):
        one_idx = col.index(1)
        trace.append(chr(ord('A') + one_idx)) 
    return trace


def convert_matrices_to_string_vectors(matrices, n_activities=8):
    string_traces = []
    for mat in matrices:
        string_trace = matrix_to_string_vector(mat, n_activities)
        string_traces.append(string_trace)
    return string_traces



def convert_matrices_to_final_predicitons(matrices):
    predictions = []
    for mat in matrices:
        prediction = probability_matrix_to_final_prediction(mat)
        predictions.append(prediction)
    return predictions

        
def probability_matrix_to_final_prediction(probability_matrix):
    probability_matrix = np.array(probability_matrix)
    indexes = probability_matrix.argmax(axis=0)
    return indexes


def convert_labels_to_final_predictions(label, n_activities = 8):
    return np.array([item[0] if len(item) == 1 else item[0] if len(item) > 1 else n_activities-1 for item in label])


def convert_labels(pred_lists):
    dic = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7}
    all_lists = []
    for list_i in pred_lists:
        temp_list = []
        for i in range(len(list_i)):
            temp_list.append(dic[list_i[i]])
        all_lists.append(temp_list)
    return all_lists
    

def argmax_of_X(X_train):
    argmax_labels =[]
    for video in X_train:
        argmax_labels.append(list(np.array(np.matrix(video).argmax(0))[0]))
    return argmax_labels

if __name__ == '__main__':

    # Models: classify to activity given preArgmax prediction (matrix)

    SignToLabel = {'A':'stir','B':'open','C':'put','D':'close','E':'take','F':'pour','G':'scoop','H':'background'}
    NumToLabel = {0:'stir',1:'open',2:'put',3:'close',4:'take',5:'pour',6:'scoop',7:'background'}
    SignToNum = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

    sandwiches_actions = DataSandwiches.actions
    sandwiches_labels = DataSandwiches.labels

    X_train_sandwiches = prepare_predictions(sandwiches_actions)
    Y_train_sandwiches = prepare_labels(sandwiches_labels)

    # a. Create list of Sequences of all GT (GT_Sequences)
    # convert letters to labels (casting to sequence), List (videos) of lists (traces) to list (videos) of sequences
    GT_Sequence_List = []
    for trace in Y_train_sandwiches:
        # GT_Sequence_List.append(Sequence([SignToLabel[trace[label]] for label in range(len(trace))]))
        GT_Sequence_List.append(Sequence([SignToNum[trace[label]] for label in range(len(trace))]))

    GTsActivities = DataSandwiches.activities
    n_activities = DataGTEA.NUM_OF_LABELS
    sequenceModel = SequenceModel.SequenceModel(GTsActivities, GT_Sequence_List, n_activities)
    logSequenceModel = LogSequenceModel.LogSequenceModel(GTsActivities, GT_Sequence_List, n_activities)


    ########## SequenceModel ##########
    x_argmax = argmax_of_X(X_train_sandwiches)

    # convert argmax list (of all SFs) to sequence
    Predictions_Sequence_List = []
    for trace in x_argmax:
        Predictions_Sequence_List.append(Sequence([trace[label] for label in range(len(trace))]))

    activity_predicted_list = []
    for seq_idx, seq in enumerate(Predictions_Sequence_List):
        activity_predicted = sequenceModel.predict(seq, seq_idx)
        activity_predicted_list.append(activity_predicted)

    results = [activity_predicted_list[pred] == GTsActivities[pred] for pred in range(len(activity_predicted_list))]
    acc = sum(results) / len(results)
    print("SequenceModel Results")
    print(f"    Activity predicted: {activity_predicted_list}")
    print(f"    Predicted right: {results}")
    print (f"   Accuracy [Trues/len]: {acc}")


    ########## logSequenceModel ##########
    activity_predicted_list = []
    for X_idx, X in enumerate(X_train_sandwiches):
        activity_predicted = logSequenceModel.predict(X, X_idx)
        activity_predicted_list.append(activity_predicted)

    results = [activity_predicted_list[pred] == GTsActivities[pred] for pred in range(len(activity_predicted_list))]    # for next time TODO: check why results are lower 3/16
    acc = sum(results) / len(results)
    print("logSequenceModel Results")
    print(f"    Activity predicted: {activity_predicted_list}")
    print(f"    Predicted right: {results}")
    print(f"   Accuracy [Trues/len]: {acc}")

    exit()



    # # Videos_len = [len(X_train_sandwiches[video][0]) for video in range(len(X_train_sandwiches))]
    # #
    # # model_sandwiches = NoiseReducerModel.NoiseReducer(DataGTEA.NUM_OF_LABELS)
    # x_argmax = argmax_of_X(X_train_sandwiches)
    # # acc_Y_before = convert_labels(Y_train_sandwiches)
    #
    # Predictions_Sequence_List = []
    # for trace in x_argmax:
    #     Predictions_Sequence_List.append(Sequence([NumToLabel[trace[label]] for label in range(len(trace))]))
    #
    # # allComparesScores = []
    # # highestScores = []
    # activity_predicted_list = []
    # for seq in Predictions_Sequence_List:
    #     activity_predicted = sequenceModel.predict(seq)
    #     activity_predicted_list.append(activity_predicted)
    #     # seq_to_GTs_scores = CompareSeqToGTs(seq, GT_Sequence_List)
    #     # allComparesScores.append(seq_to_GTs_scores)
    #     # highestScores.append(np.argmax(seq_to_GTs_scores))
    #
    # results = [activity_predicted_list[pred] == GTsActivities[pred] for pred in range(len(activity_predicted_list))]
    #
    # # TODO:
    # '''
    # a. Create list of Sequences of all GT (GT_Sequences)
    # b. Simple test - compare argmax with all GTs and plot the highest scores
    # c. Rearrange the code
    # d. Set and define how to calculate the accuracy
    # '''
    #
    # # # predicting accuracy for original neural network   # FIXedME: acc_before is less realability - adding predictions and labels before calculating
    # # final_predictions_raw_sandwiches = convert_matrices_to_final_predicitons(X_train_sandwiches)    # Eli's Code
    # # prepared_labels_sandwiches = prepare_labels_numeric(sandwiches_labels, 102, 8)
    # # for prediction, label in zip(final_predictions_raw_sandwiches, prepared_labels_sandwiches):
    # #     acc_before.append(accuracy(prediction, label))
    # #     print(accuracy(prediction, label))
    #
    # # acc_before = []
    # # for prediction, label in zip(x_argmax, acc_Y_before):
    # #     acc_before.append(accuracy(np.array(prediction), np.array(label)))
    # #
    # # print(acc_before)
    #
    #
    # # b. Simple test - compare argmax with all GTs and plot the highest scores
    # # convert num to labels (casting to sequence), List (videos) of lists (predictions) to list (videos) of sequences
    #
    #
    #
    # # X_train_sandwiches, Y_train_sandwiches = preprocess_data(X_train_sandwiches, Y_train_sandwiches)
    # model_sandwiches.train(X_train_sandwiches, Y_train_sandwiches, lr = 0.0001, n_iter = 100)
    #
    # # predicting accuracy for my algorithm
    # final_predictions_sandwiches = []
    # for trace in X_train_sandwiches:
    #     final_predictions_sandwiches.append(model_sandwiches.predict(trace))
    #
    # final_predictions_sandwiches = convert_matrices_to_final_predicitons(final_predictions_sandwiches) # argmax
    # prepared_labels_sandwiches = prepare_labels_numeric(sandwiches_labels, 102, 8) # check what this do
    # # acc_after = []
    # # for prediction, label, video_len in zip(final_predictions_sandwiches, prepared_labels_sandwiches, Videos_len):
    # #     # acc_after.append(accuracy(prediction, label, video_len))
    # #     acc_after.append(accuracy(prediction, label)) #, video_len))    # FIXME: add/remove video_len according to algo we run
    # #     # print(accuracy(prediction, label))
    # #
    # #
    # # # which activities are we missing the most? take, put? Is there a pattern?
    # # acc_compare = list(zip(acc_before, acc_after))
    # # print(acc_compare)
    #
    # # for prediction, label in zip(final_predictions_sandwiches, prepared_labels_sandwiches):
    # #     print('models prediction: ', prediction)
    # #     print('ground truth: ', label)
    #
    #
    #
    #
    #
    #
    # # # Drinks
    # #
    # # drinks_actions = Data_Drinks.actions
    # # drinks_labels = Data_Drinks.labels
    # #
    # # X_train_drinks = prepare_predictions(drinks_actions)
    # # Y_train_drinks = prepare_labels(drinks_labels)
    # #
    # # model_drinks = NoiseReducer(NUM_OF_LABELS)
    # # model_drinks.train(X_train_drinks, Y_train_drinks, lr = 0.0001, n_iter = 100)
    # #
    # #
    # # # predicting accuracy for original neural network
    # # final_predictions_raw = convert_matrices_to_final_predicitons(X_train_drinks)
    # # prepared_labels = prepare_labels_numeric(drinks_labels, 125, 8)
    # #
    # # for prediction, label in zip(final_predictions_raw, prepared_labels):
    # #     print(accuracy(prediction, label))
    # #
    # # # predicting accuracy for my algorithm
    # #
    # # final_predictions = []
    # # for trace in X_train_drinks:
    # #     final_predictions.append(model_drinks.predict(trace))
    # #
    # # final_predictions = convert_matrices_to_final_predicitons(final_predictions)
    # # prepared_labels = prepare_labels_numeric(drinks_labels, 125, 8)
    # #
    # # for prediction, label in zip(final_predictions, prepared_labels):
    # #     print(accuracy(prediction, label))
    #
    #
