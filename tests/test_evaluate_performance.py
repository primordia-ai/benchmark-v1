import os

from cristal import utils
from cristal.stats import evaluate_performance
from cristal.synthetic import GROWTH_MU, GROWTH_SCALE


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
    mixture_params = {
        "mu": {
            "high_growth": GROWTH_MU[0],
            "stable": GROWTH_MU[1],
            "time_bomb": GROWTH_MU[2]
        },
        "sigma": {
            "high_growth": GROWTH_SCALE[0],
            "stable": GROWTH_SCALE[1],
            "time_bomb": GROWTH_SCALE[2]
        }
    }

    performance_df = evaluate_performance(predictions, mixture_params, in_place=False)
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
