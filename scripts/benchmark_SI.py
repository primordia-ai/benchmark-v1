import json
import os
import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from cristal.env import PROJECT_ROOT
import sys
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from glob import glob

from cristal import synthetic

def main():
    if len(sys.argv) != 3:
        print("Usage: python bencmark_SI.py <generator_code> <extractor_code>\n Example: python benchmark_SI.py deepseek-r1:14b llama3.1:8b")
        sys.exit(1)

    generator_code = sys.argv[1]
    extractor_code = sys.argv[2]
    #cleanup the generator and extractor codes since : does not works as pathname and filename in windows
    generator_code_name = generator_code.replace(':', '_')
    extractor_code_name = extractor_code.replace(':', '_')
    # Load the ground truth data
    ground_truth = synthetic.load_indicators()

    soft_indicators_data = ground_truth.where(ground_truth['indicator_type'] == 'soft', drop=True)
    true_soft_indicators_df = pd.DataFrame(soft_indicators_data['ryg'].values, 
                                      columns=soft_indicators_data['indicator'].values,
                                      index=soft_indicators_data['asset'].values)
    print(true_soft_indicators_df.head())
    # Load the predictions from the json file in data/extracted_indicators/generator_code/extractor_code/
    predictions_path = f'../data/extracted_indicators/{generator_code_name}/{extractor_code_name}/'
    prediction_files = glob(predictions_path + '*.json')
    dfs = []
    print(f'Found {len(prediction_files)} prediction files')
    for prediction_file in prediction_files:
        asset = os.path.basename(prediction_file).split('.')[0]
        with open(prediction_file, 'r') as f:
            prediction = json.load(f)
            prediction_df = pd.DataFrame(prediction)[['value', 'indicator']]
            prediction_df = prediction_df.T #[['value', 'indicator']]
            prediction_df.columns = prediction_df.iloc[1]
            prediction_df = prediction_df[0:1]
            prediction_df.index = [asset]
            dfs.append(prediction_df)

    predictions_df = pd.concat(dfs).sort_index()
    predictions_df = predictions_df[true_soft_indicators_df.columns]
    predictions_df = predictions_df.fillna('None')
    none_counts = predictions_df.apply(lambda x: (x == 'None').sum(), axis=0)
    print("None values count per indicator:")
    print(none_counts)
    output_dir = f'../results/benchmark_SI/{generator_code_name}/{extractor_code_name}/'
    os.makedirs(output_dir, exist_ok=True)
    # Compute the confusion matrix
    confusion_matrices = []
    metrics_list = []
    # shorten the predictions_df to match the length of the true_soft_indicators_df
    true_soft_indicators_df = true_soft_indicators_df[:len(predictions_df)]
    # reindex the predictions_df to match the true_soft_indicators_df
    predictions_df = predictions_df.reindex(true_soft_indicators_df.index)

    for indicator in true_soft_indicators_df.columns:
        y_true = true_soft_indicators_df[indicator].values
        y_pred = predictions_df[indicator].values
        #make sure they are the same length
        if len(y_true) != len(y_pred):
            print(f"Length of y_true and y_pred for {indicator} are not the same")
            continue
        cm = confusion_matrix(y_true, y_pred, labels=['red', 'yellow', 'green', 'None'])
        cm_df = pd.DataFrame(cm, index=['green', 'yellow', 'red', 'None'], columns=['green', 'yellow', 'red', 'None'])
        confusion_matrices.append(cm_df)
        # Plot the confusion matrix
        plt.figure(figsize=(4, 3))
        sns.heatmap(confusion_matrices[-1], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {indicator}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{indicator}_confusion_matrix.png'))
        # Combine all accuracies into a single DataFrame
        accuracy = np.trace(confusion_matrices[-1].values) / np.sum(confusion_matrices[-1].values)
        report = classification_report(y_true, y_pred, labels=['green', 'yellow', 'red', 'None'], output_dict=True)
        mcc = matthews_corrcoef(y_true, y_pred)
        metrics_list.append({
            'Indicator': indicator,
            'Accuracy': accuracy,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1 Score': report['weighted avg']['f1-score'],
            'MCC': mcc
        })

    # Save the summary as a CSV and LaTeX table
    summary_df = pd.DataFrame(metrics_list)     
    
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)    

    summary_tex_path = os.path.join(output_dir, 'summary.tex')
    summary_tex = summary_df.to_latex(float_format="%.3f", index=False)
    summary_tex = summary_tex.replace('_', ' ')
    with open(summary_tex_path, 'w') as f:
        f.write(summary_tex)

if __name__ == '__main__':
    main()