import numpy as np
import time


def compute_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred) * 100


def compute_mean_std(accuracies):
    accuracies = np.array(accuracies)
    return np.mean(accuracies), np.std(accuracies)


def start_timer():
    return time.time()


def stop_timer(start_time):
    return time.time() - start_time
