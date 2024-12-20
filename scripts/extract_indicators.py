import pickle
import ollama
import os
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import instructor



#call a local llm and create reports for each soft indiactor in the data
# Usage: python generate_reports.py
# Output: reports for each soft indicator in the data
def get_prompt(indicator, read_path, asset_id):
    """
    Generates a detailed prompt for evaluating a specific financial indicator based on company reports.
    Args:
        indicator (str): The name of the financial indicator to evaluate.
        read_path (str): The path to the directory containing company report documents.
        asset_id (str): The identifier for the specific asset whose reports are to be read.
    Returns:
        str: A formatted prompt containing the description of the indicator, the company reports, and the indicator name.
    """

    prompt_template = f'''
        You are a financial analyst reading a report on a company's financial performance. 
        You have been given the following information:
        There are so called soft indicators that are used to evaluate the company's performance.
        The values of the indicator are represented by the colors red, yellow, and green,
        where red indicates a negative performance, yellow indicates a neutral performance, and green indicates a positive performance.
        The indicator you are evlauating is: {indicator}
        Your task is to read the business report based and infer the value of the indicator. 
        Your output should be one of the following: red, yellow, or green.
        Also give an explanation of why you chose that value.
        The description of the indicator and the company reports are given below:
        '''
    
    prompt_dict = {
        "disruptor_stagnant": """
        A measure of whether a company identified as a disruptor in its industry has plateaued or ceased innovating,
        risking its competitive advantage. This can include signs such as declining market share, reduced investment
        in research and development (R&D), or stagnation in product launches.

        - Red: Not innovative, following others
        - Yellow: Some innovative elements
        - Green: Clear industry disruptor
        """,

        "incumbent_stagnant": """
        An indicator of whether a traditional o
        r established company is failing to adapt to market changes,
        technological advancements, or evolving consumer preferences. Examples include lagging adoption of digital tools
        or inability to pivot in response to competitor advancements.

        - Red: New/marginal player
        - Yellow: Established but not dominant
        - Green: Dominant incumbent
        """,

        "turnaround_plan": """
        Assessment of the existence, feasibility, and progress of a strategic plan aimed at recovering from financial,
        operational, or reputational difficulties. This includes milestones like leadership restructuring, cost-cutting
        initiatives, or launching new revenue streams.

        - Red: No clear plan or failing execution
        - Yellow: Plan in development
        - Green: Clear, credible plan with progress
        """,

        "succession_risk": """
        Evaluation of potential risks to the company due to lack of clear succession planning for key leadership roles,
        which can impact stability and strategic continuity. Indicators include unfilled critical roles or dependency on
        a single leader without a backup plan.

        - Red: No succession plan, high dependence
        - Yellow: Basic plan in place
        - Green: Strong bench, clear succession
        """,

        "mismanagement": """
        Indicator of inefficient, unethical, or otherwise poor management practices that could negatively impact the
        company’s performance or reputation. Examples include inconsistent decision-making, lack of financial oversight,
        or ignoring stakeholder concerns.

        - Red: Poor decisions, execution issues
        - Yellow: Mixed track record
        - Green: Strong execution history
        """,

        "geopolitical_risk": """
        Assessment of the company’s exposure to risks stemming from international political tensions, trade disputes,
        sanctions, or instability in key operational regions. Specific risks may include currency fluctuations or impacts
        on supply chain operations.

        - Red: High exposure to unstable regions
        - Yellow: Moderate exposure
        - Green: Limited exposure
        """,

        "environmental_risk": """
        Measure of the company’s exposure to risks related to environmental issues, such as climate change, regulatory
        changes, or resource scarcity. Examples include penalties for non-compliance with emissions regulations or
        increased costs due to resource shortages.

        - Red: High environmental impact/risk
        - Yellow: Working on improvements
        - Green: Low impact/strong practices
        """,

        "product_moat": """
        Evaluation of the company’s ability to maintain a competitive advantage through unique products, intellectual
        property, or high barriers to entry for competitors. Signs of a strong moat include patents, exclusive
        partnerships, or economies of scale.

        - Red: No sustainable advantages
        - Yellow: Developing advantages
        - Green: Strong, sustainable moat
        """,

        "supply_chain_resilience": """
        Assessment of the robustness and adaptability of the company’s supply chain to disruptions, including natural
        disasters, geopolitical events, or supplier failures. Key factors include supplier diversification, inventory
        management strategies, and technology integration.

        - Red: Fragile, single-source dependent
        - Yellow: Some diversification
        - Green: Well-diversified, resilient
        """,

        "regulatory_pressure": """
        Indicator of the extent to which the company is subject to regulatory scrutiny or compliance challenges,
        potentially affecting its operations or financial performance. Examples include antitrust investigations, data
        privacy mandates, or sector-specific compliance burdens.

        - Red: Active investigations/issues
        - Yellow: Moderate regulatory attention
        - Green: Clean regulatory history
        """,

        "innovation_pipeline": """
        Analysis of the company’s capacity for ongoing innovation, including investments in R&D, new product
        development, and ability to adapt to technological advancements. Strong pipelines often feature consistent
        funding, cross-functional collaboration, and successful product rollouts.

        - Red: Weak pipeline, falling behind
        - Yellow: Some promising projects
        - Green: Strong, well-funded pipeline
        """,

        "customer_concentration": """
        Evaluation of risks associated with a company’s revenue being overly dependent on a small number of customers,
        which can affect bargaining power and revenue stability. Metrics include percentage of revenue from top customers
        and diversification efforts.

        - Red: Few large customers (>20%)
        - Yellow: Moderate concentration
        - Green: Well-diversified customer base
        """,

        "brand_strength": """
        Measure of the company’s brand value, reputation, and customer loyalty, which contribute to competitive
        positioning and pricing power. Factors influencing brand strength include customer satisfaction, marketing
        effectiveness, and presence in strategic markets.

        - Red: Weak or damaged brand
        - Yellow: Growing or regional brand
        - Green: Strong, valuable brand
        """,

        "digital_transformation": """
        Assessment of the company’s progress in leveraging digital technologies to enhance operations, customer
        experience, and overall competitiveness. Initiatives may include adopting cloud-based systems, automating
        workflows, and implementing AI-driven analytics.

        - Red: Lagging digital adoption
        - Yellow: Digital transition underway
        - Green: Digital leader
        """
    }

    #read the documents for the asset_id
    asset_folder = os.path.join(read_path, asset_id)
    read_documents = ""
    if os.path.exists(asset_folder):
        for filename in os.listdir(asset_folder):
            file_path = os.path.join(asset_folder, filename)
            with open(file_path, 'r') as file:
                read_documents += file.read() + "\n"

    prompt = prompt_dict[indicator]
    return prompt_template + prompt + read_documents + f"The indicator name to evaluate is: {indicator}"

def get_indicator_value(indicator_name: str, asset_id : str='', read_path : str ='../reports', model_name : str='llama3.1:8b', verbose=False)->dict:
    """
    Retrieves the value of a specified indicator for a given asset ID by generating a report using a local LLM.
    Args:
        indicator (str): The name of the indicator to retrieve.
        asset_id (str, optional): The ID of the asset for which to retrieve the indicator. Defaults to an empty string.
        read_path (str, optional): The path to the directory containing the reports. Defaults to '../reports'.
        model_name (str, optional): The name of the model to use for generating the report. Defaults to 'llama3.1:8b'.
        verbose (bool, optional): If True, prints additional information during execution. Defaults to False.
    Returns:
        dict: A dictionary containing the indicator name, value, and reason.
    """
    #has to be defined here because we need the indicator name in the Indicator class maybe this can be done in a better way?
    class Indicator(BaseModel):
        """
        Indicator model representing the performance evaluation of a company.
        Attributes:
            indicator (Literal[indicator_name]): The name of the indicator to evaluate the company's performance as given by the prompt.
            value (Literal['red', 'yellow', 'green']): The value of the indicator, represented by the colors red, yellow, and green.
            reason (str): The reason behind the indicator's value based on the company reports. This text can be long and detailed.
        """
        indicator: Literal[indicator_name] = Field(..., title="Indicator name", description="The name of the indicator to evaluate the company's performance as given by the prompt.")
        value: Literal['red', 'yellow', 'green'] = Field(..., title="Indicator value", description="The value of the indicator, represented by the colors red, yellow, and green.")
        reason: str = Field(..., title="Value reason", description="The reason behind the indicators value based on the company reports. This text can be long and detailed.")

    if verbose is True:
        tqdm.write(f'Infering value from report for asset_id {asset_id} for {indicator_name}')
    try:
        prompt = get_prompt(indicator_name, read_path, asset_id)
    except KeyError:
        tqdm.write(f'Prompt for {indicator_name} not found')
        return {'indicator': indicator_name, 'value': 'red', 'reason': f'Prompt for {indicator_name} not found'}
    except FileNotFoundError:
        tqdm.write(f'File not found for asset_id {asset_id}')
        return {'indicator': indicator_name, 'value': 'red', 'reason': f'File not found for asset_id {asset_id}'}
    
    # Use LangChain Ollama API to call a local LLM and generate a report
    #response = ollama.generate(model=model_name, prompt=prompt)
    
    client = instructor.from_openai(
                                    OpenAI(
                                        base_url="http://localhost:11434/v1",
                                        api_key="ollama",  # required, but unused
                                    ),
                                    mode=instructor.Mode.JSON,
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_model=Indicator
    )

    return json.loads(response.model_dump_json())
        
if __name__ == '__main__':
    import logging

    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    import numpy as np

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    ind = 'innovation_pipeline'#'geopolitical_risk'#'mismanagement'
    soft_indicators = ['disruptor_stagnant', 'incumbent_stagnant', 'turnaround_plan', 'succession_risk', 'mismanagement', 'geopolitical_risk', 'environmental_risk', 'product_moat', 'supply_chain_resilience', 'regulatory_pressure', 'innovation_pipeline', 'customer_concentration', 'brand_strength', 'digital_transformation']
    asset_ids = ['asset_0', 'asset_15']#, 'asset_25', 'asset_35', 'asset_45']
    test_list = {}
    for asset in tqdm(asset_ids):
        ind_list = []
        for ind in tqdm(soft_indicators, leave=False):
            indicator = get_indicator_value(indicator_name=ind, asset_id=asset, read_path='../data/reports', model_name='llama3.1:8b')
            #print(indicator)
            ind_list.append(indicator)
        test_list[asset] = ind_list
    with open('../tests/test_indicator_results.json', 'w') as f:
        json.dump(test_list, f, indent=4)