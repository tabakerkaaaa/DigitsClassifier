import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = np.float32(0)
    recall = np.float32(0)
    accuracy = np.float32(0)
    f1 = np.float32(0)
    #true_positives - количество правильных угадываний девяток, false_positives - неправильные угадывания девяток
    #false_negatives - все девятки, true_negatives - все нули
    #precision - отношение кол-ва правильных угадываний ко всем
    #recall - отношение количества правильно угаданных девяток ко всем девяткам
    true_negatives = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(prediction)):
        if (prediction[i] == ground_truth[i] and prediction[i] == False):
            true_positives += 1
        elif (prediction[i] == ground_truth[i] and prediction[i] == True):
            true_negatives += 1
        elif (prediction[i] == True and ground_truth[i] == False):
            false_positives += 1
        else:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (false_negatives + true_positives)
    accuracy = (true_positives + true_negatives) / (true_negatives + true_positives + false_negatives + false_positives)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy
