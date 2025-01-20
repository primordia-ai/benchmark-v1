import os

from alla import utils
from alla.stats import evaluate_performance


def main():
    # Seed rng
    predictions = {
        'asset1': {
            'category': {
                'high_growth': 0.1,
                'stable': 0.2,
                'time_bomb': 0.7
            },
            'growth': (
                (0.1, 0.05),
                (0.5, 0.1),
                (0.9, 0.15)
            ),
            'truth': {
                'category': 'time_bomb',
                'growth': 0.12
            }
        },
        'asset2': {
            'category': {
                'high_growth': 0.3,
                'stable': 0.4,
                'time_bomb': 0.3
            },
            'growth': (
                (0.1, 0.02),
                (0.5, 0.08),
                (0.9, 0.12)
            ),
            'truth': {
                'category': 'stable',
                'growth': 0.1
            }
        }
    }

    performance_df = evaluate_performance(predictions, in_place=False)
    print(f"MW performance_df: {performance_df}")
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
