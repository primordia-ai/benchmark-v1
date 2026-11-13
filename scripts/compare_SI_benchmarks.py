import os
import pandas as pd

def load_summary_files(directory):
    summary_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'summary.csv':
                summary_files.append(os.path.join(root, file))
    return summary_files

def generate_summary_statistics(summary_files):
    all_data = []
    for file in summary_files:
        data = pd.read_csv(file)
        data["generator_model"]=file.split('/')[-3]
        data["extractor_model"]=file.split('/')[-2]
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(combined_data)
    numeric_columns = combined_data.select_dtypes(include='number').columns
    summary_stats = combined_data.groupby(['generator_model', 'extractor_model'])[numeric_columns].agg(['mean', 'std'])
    return summary_stats

def main():
    benchmark_directory = '../results/benchmark_SI'
    summary_files = load_summary_files(benchmark_directory)
    summary_stats = generate_summary_statistics(summary_files)
    summary_stats.to_csv('../results/benchmark_SI/summary_statistics.csv')
    print("Summary statistics generated and saved to summary_statistics.csv")
    sorted_summary_stats = summary_stats.sort_values(by=('MCC', 'mean'), ascending=False)
    print(sorted_summary_stats)
    latex_table = sorted_summary_stats.to_latex(float_format="%.3f")
    latex_table = latex_table.replace('_', ':')
    with open('../results/benchmark_SI/summary_statistics.tex', 'w') as f:
        f.write(latex_table)
    print("Summary statistics LaTeX table generated and saved to summary_statistics.tex")
if __name__ == "__main__":
    main()