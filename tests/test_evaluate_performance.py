import os

from alla import utils
from alla.stats import evaluate_performance


def main():
    # Seed rng
    predictions = {
        'asset1': {
            'class_probs': [0.1, 0.7, 0.2],
            'growth_quantiles': [0.05, 0.1, 0.15],
            'alphas': [0.1, 0.5, 0.9]
        },
        'asset2': {
            'class_probs': [0.3, 0.4, 0.3],
            'growth_quantiles': [0.02, 0.08, 0.12],
            'alphas': [0.1, 0.5, 0.9]
        }
    }

    ground_truth = {
        'asset1': {
            'true_class': 1,
            'true_growth': 0.12
        },
        'asset2': {
            'true_class': 2,
            'true_growth': 0.1
        }
    }

    performance_df = evaluate_performance(predictions, ground_truth)
    # Create results directory if it doesn't exist
    results_dir = os.path.join(utils.TESTS_OUT, "test_evaluate_performance")
    os.makedirs(results_dir, exist_ok=True)

    # Save the performance DataFrame to a CSV file
    performance_df.to_csv(os.path.join(results_dir, f'performance_results.csv'), index=False)
    print(performance_df)
    print(performance_df.mean(numeric_only=True))
    # Calculate mean and standard deviation of performance metrics
    performance_summary = performance_df.describe().loc[['mean', 'std']]
    # Save the performance summary to a CSV file
    performance_summary.to_csv(os.path.join(results_dir, f'performance_summary.csv'))


if __name__ == "__main__":

    SEED = 101010

    main()
