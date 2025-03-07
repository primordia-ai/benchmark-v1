"""
LLM Utils. In particular, those for retrieving and constructing prompts
"""
import json
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from alla import synthetic as synth
from alla.env import PROJECT_ROOT

PROMPT_SRC = os.path.join(PROJECT_ROOT, "data/prompts/")


class GrowthEstimate(BaseModel):
    mean: float
    std: float


def _prob_validate(value) -> float:
    if 0 <= value <= 1.0:
        return value
    raise ValueError("Probabilities must be in range [0,1]")


class CategoryDistribution(BaseModel):
    high_growth: float = Field(
        ..., title="High Growth", description="The probability that the asset/company is a high growth investment."
    )
    stable: float = Field(
        ..., title="Stable", description="The probability that the asset/company is a stable investment."
    )
    time_bomb: float = Field(
        ..., title="Time Bomb", description="The probability that the asset/company is a time-bomb investment."
    )

    @field_validator("high_growth", mode="after")
    @classmethod
    def validate_hg(cls, value: float) -> float:
        return _prob_validate(value)

    @field_validator("stable", mode="after")
    @classmethod
    def validate_st(cls, value: float) -> float:
        return _prob_validate(value)

    @field_validator("time_bomb", mode="after")
    @classmethod
    def validate_tb(cls, value: float) -> float:
        return _prob_validate(value)

    @model_validator(mode="after")
    def normalize_probabilities(self):
        total = sum(self.dict().values())
        self.high_growth /= total
        self.stable /= total
        self.time_bomb /= total
        return self


class ExtractedIndicator(BaseModel):
    """
    Indicator color prediction based on company reports. Includes a reason why color was selected
    """
    value: Literal['red', 'yellow', 'green'] = Field(
        ..., title="Value", description="The value of the indicator, represented by the colors red, yellow, and green."
    )
    reason: str = Field(
        ..., title="Reason",
        description="The reason behind the indicator's value based on the company reports. This text can be long and detailed."
    )


class AssetEvaluationV1(BaseModel):
    category: Literal['high_growth', 'stable', 'time_bomb'] = Field(
        ..., title="Category", description="The category of the asset/company indicating its growth potential."
    )
    growth: float = Field(
        ..., title="Growth", description="The estimated growth of the asset/company."
    )


class AssetEvaluationV2(CategoryDistribution):
    pass


SOFT_INDICATOR_DESC = {
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


def prompt_se_context(indicator, report_text: str):
    """
    Generates a detailed prompt for evaluating a specific financial indicator based on company reports.
    Args:
        indicator (str): The name of the financial indicator to evaluate.
        report_text (str): The text of the company reports to analyze.
    Returns:
        str: A formatted prompt containing the description of the indicator, the company reports, and the indicator name.
    """
    prompt_template = f'''
    Below is a collection of financial reports and analyses for a certain company.
    We evaluate companies across a variety of dimensions, which we call 'indicators'.
    For each indicator, a company can have a value of 'red', 'yellow', or 'green', describing their
    performance according to the indicator.
    Your task is to consider these reports and respond with the 'value' (color) and 'reason' for choosing that color
    based on the content of the provided reports. 

    You are considering only one indicator right now, with the following description:

    Indicator: '{indicator}'
    Description: {SOFT_INDICATOR_DESC[indicator]}
    
    \n=================\nREPORTS BEGIN\n=================\n\n
    '''

    prompt = prompt_template + report_text
    prompt += "\n=================\nREPORTS END\n=================\n\n"
    prompt += ("\nPlease return your evaluation in JSON format with keys 'value' (one of red, yellow, or green) "
               "and 'reason', the reason for that choice. Please return a single evaluation for this asset.")
    return prompt


def prompt_indicator_likelihoods(select_indicator: str | None = None) -> str:
    """
    Returns the indicator likelihood matrix information, formatted for LLM consumption.
    Args:
        select_indicator (str):     Optionally specify an indicator to filter the output
    """
    ind = synth.get_indicator_dataset()
    df = ind.likelihood.to_dataframe()
    if select_indicator:
        data = df.xs(select_indicator, level='indicator')
        return f"Below is a lookup table of class and color likelihoods for the '{select_indicator}' indicator:\n\n{data.to_string()}\n"
    else:
        return f"Below is a lookup table of class and color likelihoods for company indicators:\n\n{df.to_string()}\n"


def prompt_exp1_task_deprecated(report_text: str, indicators_json: str | None):
    """
    DEPRECATED: This requests LLM point estimates, instead of mixture weights.

    Generates a detailed prompt for evaluating a specific financial indicator based on company reports.
    Args:
        report_text (str):      Reports providing information on a company.
        indicators_json (str):  Indicator data.
    Returns:
        str: A formatted prompt containing the description of the indicator, the company reports
    """
    mu_g = {cls: mu for cls, mu in zip(['high_growth', 'stable', 'time_bomb'], synth.GROWTH_MU)}
    scale_g = {cls: sigma for cls, sigma in zip(['high_growth', 'stable', 'time_bomb'], synth.GROWTH_SCALE)}

    if indicators_json:
        indicator_likelihood_prompt = prompt_indicator_likelihoods()

        text_alternatives = [
            """You have been given the following information:
                - A compilation of various performance reports in text
                - A list of financial indicators
            
            The financial indicators follow a Red/Yellow/Green construction where Red indicates a negative performance,
            yellow indicates a neutral performance, and green indicates a positive performance.
            """,
            f"""Below are the report data and indicators.
            
            REPORTS:
            {report_text}
            *************
            INDICATORS:
            {indicators_json} 
            *************
            INDICATOR LIKELIHOODS:
            {indicator_likelihood_prompt}
            """
        ]
    else:
        text_alternatives = [
            """You have been given the following information:
                - A compilation of various performance reports in text
                - A list of financial indicators
            """,
            f"""Below are the report data and indicators.
        
            REPORTS:
            {report_text}
            """,
        ]

    prompt = f"""You are a financial analyst trying to assess a company (or "asset") based on its financial performance.
    You goal is to evaluate this company across two dimensions:
        1. It's investment class, which can be one of "high_growth", "stable", or "time_bomb".
        2. Its predicted upcoming growth for the next year.
            
    The classes are self-evident: 'high_growth' companies are promising investments; 'time_bomb' companies have 
    problematic outlooks; and 'stable' companies are the moderate.
    ---
    {text_alternatives[0]}
    
    You are also given some statistical parameters that underlie company growth rates by class, 
    which parametrize a Gaussian form:
        mu_growth: {mu_g},
        sigma_growth: {scale_g}
    Keep in mind, that just because an asset may belong to a certain investment class, does not mean its
    growth is the mean from mu_growth -- You'll have to consider the reports and indicators to further
    inform the exact growth value
    ---
    {text_alternatives[1]}
    ---
    You have received all the information you need.
    Please return your evaluation in JSON format with keys "category" (str) and "growth" (float).
    """

    return prompt


def prompt_exp1_task(report_text: str, indicators_json: str | None):
    """
    Generates a detailed prompt for evaluating a specific financial indicator based on company reports.

    Args:
        report_text (str):      Reports providing information on a company.
        indicators_json (str):  Indicator data.
    Returns:
        str: A formatted prompt containing the description of the indicator, the company reports
    """
    mu_g = {cls: mu for cls, mu in zip(['high_growth', 'stable', 'time_bomb'], synth.GROWTH_MU)}
    scale_g = {cls: sigma for cls, sigma in zip(['high_growth', 'stable', 'time_bomb'], synth.GROWTH_SCALE)}

    if indicators_json:
        indicator_likelihood_prompt = prompt_indicator_likelihoods()

        text_alternatives = [
            """You have been given the following information:
                - A compilation of various performance reports in text
                - A list of financial indicators

            The financial indicators follow a Red/Yellow/Green construction where Red indicates a negative performance,
            yellow indicates a neutral performance, and green indicates a positive performance.
            """,
            f"""Below are the report data and indicators.

            REPORTS:
            {report_text}
            *************
            INDICATORS:
            {indicators_json} 
            *************
            INDICATOR LIKELIHOODS:
            {indicator_likelihood_prompt}
            """
        ]
    else:
        text_alternatives = [
            """You have been given the following information:
                - A compilation of various performance reports in text
                - A list of financial indicators
            """,
            f"""Below is the report data.

            REPORTS:
            {report_text}
            """,
        ]

    prompt = f"""You are a financial analyst trying to assess a company (or "asset") based on its financial performance.
    You goal is to evaluate this company to determine its category (class) likelihood.
    
    There are three categories: "high_growth", "stable", and "time_bomb".
    The classes are self-evident: 'high_growth' companies are promising investments; 'time_bomb' companies have 
    problematic outlooks; and 'stable' companies are the moderate.
    
    Your output must be a distribution of probabilities corresponding to your category beliefs for the given asset.
    So, it will be three probabilities that sum to 1.
    ---
    {text_alternatives[0]}

    ---
    {text_alternatives[1]}
    
    ---
    You have received all the information you need.
    Please return your evaluation in JSON format as:
     
    "high_growth": <float>, 
    "stable": <float>, 
    "time_bomb": <float>
    """

    return prompt


def prompt_critic_exp1(previous_result: str) -> str:
    """
    Prompt from the critic that prompts the LLM to update its last prediction
    """
    return f"""
    Your latest predicted distribution was: {previous_result}
    
    Please consider all the details in the reports critically, and update your prediction. 
    This is not to say your previous prediction is inaccurate (though it might be), we just want to be as accurate as possible.

    Please return your updated prediction in JSON format as:
     
    "high_growth": <float>, 
    "stable": <float>, 
    "time_bomb": <float>
    """


def prompt_exp1_prolog() -> str:
    """
    Text prolog describing the aim of the first experiment and other important details.
    """
    pass


def prompt_codegen_inference_with_measurement() -> str:
    """
    Prompt an LLM to generate an inference algorithm
    """
    pass


def load_extracted_indicators(json_path, keys: list[str] | None = None) -> list[dict]:
    """
    Load the extracted indicators from the JSON file.

    Args:
        json_path (str): The path to the JSON file containing the extracted indicators.
        keys (list[str]): The keys to extract from the JSON file. Any of ["value", "indicator", "reason"]
    """
    with open(json_path, 'r') as f:
        result = json.load(f)

    keys = keys or ["value", "indicator", "reason"]

    return [
        {
            key: r[key] for key in keys
        } for r in result
    ]
