import os
import pickle

import ollama
from tqdm import tqdm

from alla.llm import utils as llm_utils


# call a local llm and create reports for each soft indiactor in the data
# Usage: python generate_reports.py
# Output: reports for each soft indicator in the data
def get_prompt(indicator, value):
    """
    Generates a prompt for a financial report based on the given indicator and value.
    Args:
        indicator (str): The key representing the type of financial indicator.
        value (str): The value associated with the financial indicator.
    Returns:
        str: A formatted prompt string based on the indicator and value.
    Raises:
        KeyError: If the provided indicator is not found in the prompt dictionary.
    """

    prompt_template = '''
        You are a financial reporter writing a report on a company's financial performance. 
        You have been given the following information:
        There are so called soft indicators that are used to evaluate the company's performance.
        The values of the indicator are represented by the colors red, yellow, and green,
        where red indicates a negative performance, yellow indicates a neutral performance, and green indicates a positive performance.
        Your task is to write a business report based on the given indicator and its value. 
        Do not use the actual indicator name anywhere the report.
        Also do not use the actual value of the indicator in the report.
        Do not give your report a headline that might give away the indicator name.
        Format the report in markdown.
        The description and value of the indicator are given below:
        '''

    prompt = llm_utils.SOFT_INDICATOR_DESC[indicator] + f"""The indicator value is: {value}"""
    return prompt_template + prompt


def generate_report(
    indicator: str, value: str, asset_id: str = '', save_path: str = '../reports', model_name: str = 'llama3.1:8b',
    verbose=False
) -> str:
    """
    Generates a report based on the given indicator and value using a specified model.
    Args:
        indicator (str): The indicator for which the report is to be generated.
        value (str): The value associated with the indicator.
        asset_id (str, optional): The asset ID for which the report is being generated. Defaults to ''.
        save_path (str, optional): The path where the report will be saved. Defaults to 'reports'.
        model_name (str, optional): The name of the model to be used for generating the report. Defaults to 'llama3.1:8b'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
        str: The generated report as a string.
    Raises:
        KeyError: If the prompt for the given indicator is not found.
    """
    if verbose == True:
        tqdm.write(f'Generating report for {indicator} with value {value}')
    try:
        prompt = get_prompt(indicator, value)
    except KeyError:
        tqdm.write(f'Prompt for {indicator} not found')
        return f'Prompt for {indicator} not found'

    # Use LangChain Ollama API to call a local LLM and generate a report
    response = ollama.generate(model=model_name, prompt=prompt)
    report = response['response']
    # Save the report to the specified path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, asset_id)):
        os.makedirs(os.path.join(save_path, asset_id))

    report_path = os.path.join(save_path, f'{asset_id}/{indicator}_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.write(report)
    return report


if __name__ == '__main__':
    import logging

    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    import numpy as np

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # Read the pkl file
    test = False
    if test == True:
        generate_report(
            indicator='turnaround_plan', value='green', asset_id='test_asset', save_path='../data/reports',
            model_name='llama3.1:8b'
        )

    else:
        ground_truth_path = '../data/ground-truth/stock_indicators.pkl'
        with open(ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)

        soft_indicators_data = ground_truth.where(ground_truth['indicator_type'] == 'soft', drop=True)

        for asset_id, soft_indicator_values in tqdm(
                zip(soft_indicators_data['asset'].values, soft_indicators_data['ryg'].values), desc="Assets",
                leave=True, position=0
        ):
            data = dict(zip(soft_indicators_data['indicator'].values, soft_indicator_values))
            for indicator, value in tqdm(
                    data.items(), desc="generating reports for indicators", position=1, leave=False
            ):
                generate_report(indicator=indicator, value=value, asset_id=asset_id)
