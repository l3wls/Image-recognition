# Import the main experiment function
from src.runner import run_experiment

# Define combinations of datasets and algorithms to test
experiments = [
    ("digits", "naive_bayes"),
    ("digits", "perceptron"),
    ("faces", "naive_bayes"),
    ("faces", "perceptron"),
]

# Loop over different feature extraction methods
for feature_type in ("pixel", "grid"):

    # Loop over each dataset and algorithm combination
    for dataset, algorithm in experiments:

        # Print header to clearly separate experiment results
        print(f"\n{'='*56}")
        print(f"  {dataset.upper()} | {algorithm.upper()} | {feature_type.upper()} FEATURES")
        print(f"{'='*56}")

        # Run the experiment with selected configuration
        run_experiment(dataset, algorithm, feature_type)