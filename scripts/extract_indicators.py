import pickle
import ollama
import os
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json
import instructor

class Indicator(BaseModel):
    indicator: str = Field(..., title="Indicator name", description="The name of the indicator to evaluate the company's performance as given by the prompt.")
    value: str = Field(..., title="Indicator value", description="The value of the indicator, represented by the colors red, yellow, and green.")
    reason: str = Field(..., title="Value reason", description="The reason behind the indicators value based on the company reports. This text can be long and detailed.")


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
        Also give a detailed explanation of why you chose that value.
        The description of the indicator and the company reports are given below:
        '''
    
    prompt_dict = {
        "disruptor_stagnant": """
        A measure of whether a company identified as a disruptor in its industry has plateaued or ceased innovating,
        risking its competitive advantage. This can include signs such as declining market share, reduced investment
        in research and development (R&D), or stagnation in product launches.
        """,

        "incumbent_stagnant": """
        An indicator of whether a traditional or established company is failing to adapt to market changes,
        technological advancements, or evolving consumer preferences. Examples include lagging adoption of digital tools
        or inability to pivot in response to competitor advancements.
        """,

        "turnaround_plan": """
        Assessment of the existence, feasibility, and progress of a strategic plan aimed at recovering from financial,
        operational, or reputational difficulties. This includes milestones like leadership restructuring, cost-cutting
        initiatives, or launching new revenue streams.
        """,

        "succession_risk": """
        Evaluation of potential risks to the company due to lack of clear succession planning for key leadership roles,
        which can impact stability and strategic continuity. Indicators include unfilled critical roles or dependency on
        a single leader without a backup plan.
        """,

        "mismanagement": """
        Indicator of inefficient, unethical, or otherwise poor management practices that could negatively impact the
        company’s performance or reputation. Examples include inconsistent decision-making, lack of financial oversight,
        or ignoring stakeholder concerns.
        """,

        "geopolitical_risk": """
        Assessment of the company’s exposure to risks stemming from international political tensions, trade disputes,
        sanctions, or instability in key operational regions. Specific risks may include currency fluctuations or impacts
        on supply chain operations.
        """,

        "environmental_risk": """
        Measure of the company’s exposure to risks related to environmental issues, such as climate change, regulatory
        changes, or resource scarcity. Examples include penalties for non-compliance with emissions regulations or
        increased costs due to resource shortages.
        """,

        "product_moat": """
        Evaluation of the company’s ability to maintain a competitive advantage through unique products, intellectual
        property, or high barriers to entry for competitors. Signs of a strong moat include patents, exclusive
        partnerships, or economies of scale.
        """,

        "supply_chain_resilience": """
        Assessment of the robustness and adaptability of the company’s supply chain to disruptions, including natural
        disasters, geopolitical events, or supplier failures. Key factors include supplier diversification, inventory
        management strategies, and technology integration.
        """,

        "regulatory_pressure": """
        Indicator of the extent to which the company is subject to regulatory scrutiny or compliance challenges,
        potentially affecting its operations or financial performance. Examples include antitrust investigations, data
        privacy mandates, or sector-specific compliance burdens.
        """,

        "innovation_pipeline": """
        Analysis of the company’s capacity for ongoing innovation, including investments in R&D, new product
        development, and ability to adapt to technological advancements. Strong pipelines often feature consistent
        funding, cross-functional collaboration, and successful product rollouts.
        """,

        "customer_concentration": """
        Evaluation of risks associated with a company’s revenue being overly dependent on a small number of customers,
        which can affect bargaining power and revenue stability. Metrics include percentage of revenue from top customers
        and diversification efforts.
        """,

        "brand_strength": """
        Measure of the company’s brand value, reputation, and customer loyalty, which contribute to competitive
        positioning and pricing power. Factors influencing brand strength include customer satisfaction, marketing
        effectiveness, and presence in strategic markets.
        """,

        "digital_transformation": """
        Assessment of the company’s progress in leveraging digital technologies to enhance operations, customer
        experience, and overall competitiveness. Initiatives may include adopting cloud-based systems, automating
        workflows, and implementing AI-driven analytics.
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

def get_indicator_value(indicator: str, asset_id : str='', read_path : str ='../reports', model_name : str='llama3.1:8b', verbose=False)->dict:
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

    if verbose is True:
        tqdm.write(f'Infering value from report for asset_id {asset_id} for {indicator}')
    try:
        prompt = get_prompt(indicator, read_path, asset_id)
    except KeyError:
        tqdm.write(f'Prompt for {indicator} not found')
        return {'indicator': indicator, 'value': 'red', 'reason': f'Prompt for {indicator} not found'}
    except FileNotFoundError:
        tqdm.write(f'File not found for asset_id {asset_id}')
        return {'indicator': indicator, 'value': 'red', 'reason': f'File not found for asset_id {asset_id}'}
    
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
    ind = 'mismanagement'
    indicator = get_indicator_value(indicator=ind, asset_id='asset_0', read_path='../reports', model_name='llama3.1:8b')
    print(indicator)