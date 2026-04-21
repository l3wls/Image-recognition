from src.runner import run_experiment

experiments = [
    ("digits", "naive_bayes"),
    ("digits", "perceptron"),
    ("faces", "naive_bayes"),
    ("faces", "perceptron"),
]

for feature_type in ("pixel","grid"):
    for dataset, algorithm in experiments:
        print(f"\n{'='*56}")
        print(f"  {dataset.upper()} | {algorithm.upper()} | {feature_type.upper()} FEATURES")
        print(f"{'='*56}")
        run_experiment(dataset, algorithm, feature_type)
