import numpy as np
import random
import time

from src.data_loader import load_digit_training, load_digit_test, load_face_training, load_face_test
from src.features.pixel_features import extract_pixel_features
from src.features.counting_features import extract_counting_features
from src.features.grid_features import extract_grid_features
from src.naive_bayes import NaiveBayes
from src.perceptron import Perceptron
from src.evaluator import compute_accuracy, compute_mean_std, start_timer, stop_timer


def run_experiment(dataset, algorithm, feature_type, n_trials=5):
    if dataset == "digits":
        train_images, train_labels = load_digit_training()
        test_images, test_labels = load_digit_test()
    else:
        train_images, train_labels = load_face_training()
        test_images, test_labels = load_face_test()

    if feature_type == "pixel":
        extractor = extract_pixel_features
    elif feature_type == "grid":
        extractor = extract_grid_features
    else:
        extractor = extract_counting_features

    all_train_features = np.array([extractor(img) for img in train_images])
    all_train_labels = np.array(train_labels)

    test_features = np.array([extractor(img) for img in test_images])
    test_labels_arr = np.array(test_labels)

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_train = len(all_train_labels)

    print(f"\nDataset={dataset}  Algorithm={algorithm}  Features={feature_type}")
    print(f"{'% Data':<10} {'Acc Mean':>10} {'Acc Std':>10} {'Time Mean':>12} {'Time Std':>10}")
    print("-" * 56)

    for pct in percentages:
        trial_accuracies = []
        trial_runtimes = []

        sample_size = max(1, int(n_train * pct))
        indices = list(range(n_train))

        for _ in range(n_trials):
            sampled = random.sample(indices, sample_size)
            X_sample = all_train_features[sampled]
            y_sample = all_train_labels[sampled]

            model = NaiveBayes() if algorithm == "naive_bayes" else Perceptron()

            t_start = start_timer()
            model.fit(X_sample, y_sample)
            elapsed = stop_timer(t_start)

            y_pred = model.predict(test_features)
            acc = compute_accuracy(test_labels_arr, y_pred)

            trial_accuracies.append(acc)
            trial_runtimes.append(elapsed)

        acc_mean, acc_std = compute_mean_std(trial_accuracies)
        time_mean, time_std = compute_mean_std(trial_runtimes)

        print(f"{pct:<10.0%} {acc_mean:>9.2f}% {acc_std:>9.2f}% {time_mean:>11.4f}s {time_std:>9.4f}s")
