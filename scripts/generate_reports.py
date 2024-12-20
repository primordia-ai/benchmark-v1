import pickle
import ollama
import os
from tqdm import tqdm

#call a local llm and create reports for each soft indiactor in the data
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
        Do not give your report a headline that might give awway the indicator name.
        Format the report in markdown.
        The description and value of the indicator are given below:
        '''
    prompt_dict = {
    "disruptor_stagnant": """A measure of whether a company identified as a disruptor in its industry has plateaued or ceased innovating, risking its competitive advantage. """,
    "incumbent_stagnant": """An indicator of whether a traditional or established company is failing to adapt to market changes, technological advancements, or evolving consumer preferences.""",
    "turnaround_plan": """Assessment of the existence, feasibility, and progress of a strategic plan aimed at recovering from financial, operational, or reputational difficulties.""",
    "succession_risk": """Evaluation of potential risks to the company due to lack of clear succession planning for key leadership roles, which can impact stability and strategic continuity.""",
    "mismanagement": """Indicator of inefficient, unethical, or otherwise poor management practices that could negatively impact the company’s performance or reputation.""",
    "geopolitical_risk": """Assessment of the company’s exposure to risks stemming from international political tensions, trade disputes, sanctions, or instability in key operational regions.""",
    "environmental_risk": """Measure of the company’s exposure to risks related to environmental issues, such as climate change, regulatory changes, or resource scarcity.""",
    "product_moat": """Evaluation of the company’s ability to maintain a competitive advantage through unique products, intellectual property, or high barriers to entry for competitors.""",
    "supply_chain_resilience": """Assessment of the robustness and adaptability of the company’s supply chain to disruptions, including natural disasters, geopolitical events, or supplier failures.""",
    "regulatory_pressure": """Indicator of the extent to which the company is subject to regulatory scrutiny or compliance challenges, potentially affecting its operations or financial performance.""",
    "innovation_pipeline": """Analysis of the company’s capacity for ongoing innovation, including investments in R&D, new product development, and ability to adapt to technological advancements.""",
    "customer_concentration": """Evaluation of risks associated with a company’s revenue being overly dependent on a small number of customers, which can affect bargaining power and revenue stability.""",
    "brand_strength": """Measure of the company’s brand value, reputation, and customer loyalty, which contribute to competitive positioning and pricing power.""",
    "digital_transformation": """Assessment of the company’s progress in leveraging digital technologies to enhance operations, customer experience, and overall competitiveness."""
}

    prompt = prompt_dict[indicator] + f"""The indicator value is: {value}"""
    return prompt_template + prompt

def generate_report(indicator: str, value: str, asset_id : str='', save_path : str ='../reports', model_name : str='llama3.1:8b')->str:
    """
    Generates a report based on the given indicator and value using a specified model.
    Args:
        indicator (str): The indicator for which the report is to be generated.
        value (str): The value associated with the indicator.
        asset_id (str, optional): The asset ID for which the report is being generated. Defaults to ''.
        save_path (str, optional): The path where the report will be saved. Defaults to 'reports'.
        model_name (str, optional): The name of the model to be used for generating the report. Defaults to 'llama3.1:8b'.
    Returns:
        str: The generated report as a string.
    Raises:
        KeyError: If the prompt for the given indicator is not found.
    """
    
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
    # Read the pkl file
    test=False
    if test==True:
        generate_report(indicator='turnaround_plan', value='green', asset_id='test_asset', save_path='../reports', model_name='llama3.1:8b')

    else:
        ground_truth_path = '../ground-truth/stock_indicators.pkl'
        with open(ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)

        soft_indicators_data = ground_truth.where(ground_truth['indicator_type'] == 'soft', drop=True)

        for asset_id, soft_indicator_values in tqdm(zip(soft_indicators_data['asset'].values[:2], soft_indicators_data['ryg'].values[:2])):
            data = dict(zip(soft_indicators_data['indicator'].values, soft_indicator_values))
            for indicator, value in tqdm(data.items()):
                generate_report(indicator=indicator, value=value, asset_id=asset_id)
        
