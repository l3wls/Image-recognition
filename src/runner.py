import numpy as np
import random
import time

# Import data loading functions for digits and faces datasets
from src.data_loader import load_digit_training, load_digit_test, load_face_training, load_face_test

# Import feature extraction methods
from src.features.pixel_features import extract_pixel_features
from src.features.grid_features import extract_grid_features

# Import machine learning models
from src.naive_bayes import NaiveBayes
from src.perceptron import Perceptron

# Import evaluation utilities
from src.evaluator import compute_accuracy, compute_mean_std, start_timer, stop_timer


def run_experiment(dataset, algorithm, feature_type, n_trials=5):
    # LOAD THE  DATA 

    # Select dataset (digits or faces)
    if dataset == "digits":
        train_images, train_labels = load_digit_training()
        test_images, test_labels = load_digit_test()
    else:
        train_images, train_labels = load_face_training()
        test_images, test_labels = load_face_test()

    #  SELECT FEATURE EXTRACTION

    # Choose feature extraction method
    if feature_type == "pixel":
        extractor = extract_pixel_features
    elif feature_type == "grid":
        extractor = extract_grid_features

    # Convert training images into numerical feature vectors
    all_train_features = np.array([extractor(img) for img in train_images])
    all_train_labels = np.array(train_labels)

    # Convert test images into feature vectors
    test_features = np.array([extractor(img) for img in test_images])
    test_labels_arr = np.array(test_labels)

   

    # Different percentages of training data to test performance
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Total number of training samples
    n_train = len(all_train_labels)

    # Print experiment configuration
    print(f"\nDataset={dataset}  Algorithm={algorithm}  Features={feature_type}")
    print(f"{'% Data':<10} {'Acc Mean':>10} {'Acc Std':>10} {'Time Mean':>12} {'Time Std':>10}")
    print("-" * 56)

    #MAIN EXPERIMENT LOOP 

    # Loop over different dataset sizes
    for pct in percentages:
        trial_accuracies = []
        trial_runtimes = []

        # Determine number of samples for this percentage
        sample_size = max(1, int(n_train * pct))

        # Create list of indices for sampling
        indices = list(range(n_train))

        # Run multiple trials for averaging results
        for _ in range(n_trials):

            # Randomly sample training data
            sampled = random.sample(indices, sample_size)
            X_sample = all_train_features[sampled]
            y_sample = all_train_labels[sampled]

            # Select model (Naive Bayes or Perceptron)
            model = NaiveBayes() if algorithm == "naive_bayes" else Perceptron()

            #  TRAINING 

            # Start timing
            t_start = start_timer()

            # Train model on sampled data
            model.fit(X_sample, y_sample)

            # Stop timing
            elapsed = stop_timer(t_start)

            #  TESTING 

            # Predict labels for test data
            y_pred = model.predict(test_features)

            # Compute accuracy of predictions
            acc = compute_accuracy(test_labels_arr, y_pred)

            # Store results for this trial
            trial_accuracies.append(acc)
            trial_runtimes.append(elapsed)

        #  COMPUTE the stats

        # Compute mean and standard deviation of accuracy
        acc_mean, acc_std = compute_mean_std(trial_accuracies)

        # Compute mean and standard deviation of runtime
        time_mean, time_std = compute_mean_std(trial_runtimes)

        # PRINT the results 

        print(f"{pct:<10.0%} {acc_mean:>9.2f}% {acc_std:>9.2f}% {time_mean:>11.4f}s {time_std:>9.4f}s")