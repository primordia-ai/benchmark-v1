"""
LLM Utils. In particular, those for retrieving and constructing prompts
"""
import os
import json

from alla.utils import PROJECT_ROOT

PROMPT_SRC = os.path.join(PROJECT_ROOT, "src/llm/prompts/")

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
        You are a financial analyst reading a report on a company's financial performance. 
        You have been given the following information:
        There are so called soft indicators that are used to evaluate the company's performance.
        The values of the indicator are represented by the colors red, yellow, and green,
        where red indicates a negative performance, yellow indicates a neutral performance, and green indicates a positive performance.
        The indicator you are evaluating is: '{indicator}'
        Your task is to read the business reports and infer the value of the indicator. 
        Your output should be one of the following: red, yellow, or green.
        Also give an explanation of why you chose that value.
        The description of the indicator and the company reports are given below:
        '''

    indicator_desc = SOFT_INDICATOR_DESC[indicator]

    return prompt_template + indicator_desc + report_text + f"\nThe indicator name to evaluate is: {indicator}"


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
