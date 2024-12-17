import numpy as np

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
from typing import List, Dict, Any

import xarray as xr
import numpy as np

# Core independent indicators with more realistic bounds
INDEPENDENT_INDICATOR_INTERVALS = {
    # Profitability fundamentals (even troubled companies rarely go below -0.2)
    'gross_margin': {
        'red': (-0.2, 0.15),
        'yellow': (0.15, 0.40),
        'green': (0.40, 0.80)
    },
    'operating_margin': {
        'red': (-0.2, 0.05),
        'yellow': (0.05, 0.20),
        'green': (0.20, 0.45)
    },

    # Capital structure and efficiency
    'asset_turnover': {
        'red': (0.2, 0.5),
        'yellow': (0.5, 2.0),
        'green': (2.0, 4.0)
    },
    'leverage': {
        'red': (2.0, 5.0),
        'yellow': (0.5, 2.0),
        'green': (0.1, 0.5)
    },

    # Growth metrics (more realistic bounds)
    'revenue_growth': {
        'red': (-0.3, 0.0),
        'yellow': (0.0, 0.20),
        'green': (0.20, 0.50)
    },
    'ebitda_growth': {
        'red': (-0.4, -0.05),
        'yellow': (-0.05, 0.15),
        'green': (0.15, 0.60)
    },

    # Operational efficiency
    'working_capital_ratio': {
        'red': (0.5, 1.0),
        'yellow': (1.0, 2.0),
        'green': (2.0, 4.0)
    },
    'reinvestment_rate': {
        'red': (0.0, 0.15),
        'yellow': (0.15, 0.75),
        'green': (0.75, 1.2)
    }
}

# Define default tolerance thresholds (tighter for core metrics, looser for derived ones)
DEFAULT_TOLERANCES = {
    # Core profitability metrics - tight tolerance
    'operating_margin': 0.05,  # 5% relative error
    'gross_margin': 0.05,
    'net_profit_margin': 0.05,

    # Core balance sheet metrics - tight tolerance
    'leverage': 0.05,
    'working_capital_ratio': 0.10,

    # Core return metrics - moderate tolerance
    'roic': 0.10,
    'roe': 0.10,
    'cfroi': 0.15,

    # Growth metrics - looser tolerance due to compounding effects
    'revenue_growth': 0.15,
    'fcf_growth': 0.20,
    'ebitda_growth': 0.20,

    # Efficiency metrics - moderate tolerance
    'asset_turnover': 0.15,
    'reinvestment_rate': 0.15,

    # Market-based metrics - looser tolerance due to interdependencies
    'pe_ratio': 0.20,
    'ev_ebitda': 0.20,
    'pb_ratio': 0.20,

    # Derived metrics - looser tolerance
    'eva_percent': 0.25,

    # Cash conversion - moderate tolerance
    'cash_conversion_cycle': 0.15
}

FINANCIAL_LIKELIHOOD_MATRIX = {
    'high_growth': {
        # Core profitability
        'gross_margin': {'red': 0.1, 'green': 0.6},  # Often strong margins but can sacrifice for growth
        'operating_margin': {'red': 0.3, 'green': 0.4},  # Variable due to growth investments

        # Efficiency and structure
        'asset_turnover': {'red': 0.2, 'green': 0.4},  # Moderate efficiency while scaling
        'leverage': {'red': 0.3, 'green': 0.4},  # Often uses some leverage to fuel growth

        # Growth rates
        'revenue_growth': {'red': 0.05, 'green': 0.9},  # Defining characteristic - high growth
        'ebitda_growth': {'red': 0.1, 'green': 0.6},  # Strong but may lag revenue growth

        # Operations
        'working_capital_ratio': {'red': 0.4, 'green': 0.3},  # Often tight on working capital
        'reinvestment_rate': {'red': 0.1, 'green': 0.8}  # Heavy reinvestment in growth
    },

    'stable': {
        # Core profitability
        'gross_margin': {'red': 0.1, 'green': 0.5},  # Consistent, healthy margins
        'operating_margin': {'red': 0.1, 'green': 0.5},  # Stable operational efficiency

        # Efficiency and structure
        'asset_turnover': {'red': 0.1, 'green': 0.4},  # Consistent asset efficiency
        'leverage': {'red': 0.1, 'green': 0.6},  # Conservative capital structure

        # Growth rates
        'revenue_growth': {'red': 0.1, 'green': 0.1},  # Modest, steady growth
        'ebitda_growth': {'red': 0.1, 'green': 0.3},  # Consistent earnings growth

        # Operations
        'working_capital_ratio': {'red': 0.1, 'green': 0.5},  # Strong working capital management
        'reinvestment_rate': {'red': 0.2, 'green': 0.1}  # Maintenance level investment
    },

    'time_bomb': {
        # Core profitability
        'gross_margin': {'red': 0.5, 'green': 0.1},  # Deteriorating margins
        'operating_margin': {'red': 0.6, 'green': 0.1},  # Poor operational efficiency

        # Efficiency and structure
        'asset_turnover': {'red': 0.6, 'green': 0.1},  # Poor asset utilization
        'leverage': {'red': 0.8, 'green': 0.1},  # Excessive leverage

        # Growth rates
        'revenue_growth': {'red': 0.6, 'green': 0.1},  # Declining revenues
        'ebitda_growth': {'red': 0.6, 'green': 0.1},  # Deteriorating earnings

        # Operations
        'working_capital_ratio': {'red': 0.7, 'green': 0.1},  # Liquidity problems
        'reinvestment_rate': {'red': 0.7, 'green': 0.1}  # Underinvestment
    }
}

SOFT_LIKELIHOOD_MATRIX = {
    'high_growth': {
        'disruptor_stagnant': {'red': 0.1, 'green': 0.7},  # Often disruptors
        'incumbent_stagnant': {'red': 0.6, 'green': 0.2},  # Rarely incumbents
        'turnaround_plan': {'red': 0.7, 'green': 0.2},  # Rarely need turnaround
        'succession_risk': {'red': 0.3, 'green': 0.4},  # Mixed succession risk
        'mismanagement': {'red': 0.2, 'green': 0.5},  # Usually well-managed
        'geopolitical_risk': {'red': 0.2, 'green': 0.4},  # Moderate geo risk
        'environmental_risk': {'red': 0.2, 'green': 0.5},  # Often environmentally conscious
        'product_moat': {'red': 0.2, 'green': 0.6},  # Strong moats
        'supply_chain_resilience': {'red': 0.4, 'green': 0.3},  # Can be fragile while scaling
        'regulatory_pressure': {'red': 0.4, 'green': 0.3},  # Often face scrutiny
        'innovation_pipeline': {'red': 0.1, 'green': 0.7},  # Strong innovation
        'customer_concentration': {'red': 0.3, 'green': 0.4},  # Moderate concentration
        'brand_strength': {'red': 0.3, 'green': 0.5},  # Building strong brands
        'digital_transformation': {'red': 0.1, 'green': 0.8}  # Digital native
    },

    'stable': {
        'disruptor_stagnant': {'red': 0.6, 'green': 0.2},  # Rarely disruptors
        'incumbent_stagnant': {'red': 0.2, 'green': 0.6},  # Often incumbents
        'turnaround_plan': {'red': 0.7, 'green': 0.2},  # Rarely need turnaround
        'succession_risk': {'red': 0.2, 'green': 0.6},  # Good succession planning
        'mismanagement': {'red': 0.2, 'green': 0.6},  # Well-managed
        'geopolitical_risk': {'red': 0.2, 'green': 0.5},  # Usually well-hedged
        'environmental_risk': {'red': 0.3, 'green': 0.4},  # Mixed environmental risk
        'product_moat': {'red': 0.2, 'green': 0.6},  # Established moats
        'supply_chain_resilience': {'red': 0.2, 'green': 0.6},  # Robust supply chains
        'regulatory_pressure': {'red': 0.2, 'green': 0.5},  # Generally compliant
        'innovation_pipeline': {'red': 0.3, 'green': 0.4},  # Moderate innovation
        'customer_concentration': {'red': 0.2, 'green': 0.6},  # Well diversified
        'brand_strength': {'red': 0.2, 'green': 0.6},  # Strong established brands
        'digital_transformation': {'red': 0.3, 'green': 0.4}  # Mixed digital progress
    },

    'time_bomb': {
        'disruptor_stagnant': {'red': 0.8, 'green': 0.1},  # Not disruptive
        'incumbent_stagnant': {'red': 0.3, 'green': 0.5},  # Often struggling incumbents
        'turnaround_plan': {'red': 0.6, 'green': 0.2},  # Poor turnaround execution
        'succession_risk': {'red': 0.7, 'green': 0.1},  # High succession risk
        'mismanagement': {'red': 0.7, 'green': 0.1},  # Often mismanaged
        'geopolitical_risk': {'red': 0.5, 'green': 0.2},  # High geo risk
        'environmental_risk': {'red': 0.6, 'green': 0.2},  # High environmental risk
        'product_moat': {'red': 0.7, 'green': 0.1},  # Weak moat
        'supply_chain_resilience': {'red': 0.6, 'green': 0.2},  # Supply chain issues
        'regulatory_pressure': {'red': 0.6, 'green': 0.2},  # Regulatory troubles
        'innovation_pipeline': {'red': 0.7, 'green': 0.1},  # Weak innovation
        'customer_concentration': {'red': 0.6, 'green': 0.2},  # Often concentrated
        'brand_strength': {'red': 0.6, 'green': 0.2},  # Weakening brands
        'digital_transformation': {'red': 0.7, 'green': 0.1}  # Digital laggard
    }
}

SOFT_INDICATOR_COSTS = {
    'disruptor_stagnant': 6,  # Abstract market position.
    'incumbent_stagnant': 3,  # Clearer in performance reviews.
    'turnaround_plan': 7,  # Strategic execution subtle.
    'succession_risk': 9,  # Hardest, requires leadership signals.
    'mismanagement': 8,  # Implied via scandals or failures.
    'geopolitical_risk': 2,  # Explicit in news and sanctions.
    'environmental_risk': 4,  # Found in ESG reports.
    'product_moat': 7,  # Competitive advantage is subtle.
    'supply_chain_resilience': 8,  # Needs supplier analysis.
    'regulatory_pressure': 5,  # Clear in regulatory actions.
    'innovation_pipeline': 6,  # Patent and R&D mentions.
    'customer_concentration': 7,  # Key clients often not clear.
    'brand_strength': 4,  # Sentiment and brand mentions.
    'digital_transformation': 5  # Explicit in strategy docs.
}

# Combine the above
INDICATOR_LIKELIHOODS = {
    'high_growth': {**FINANCIAL_LIKELIHOOD_MATRIX['high_growth'], **SOFT_LIKELIHOOD_MATRIX['high_growth']},
    'stable': {**FINANCIAL_LIKELIHOOD_MATRIX['stable'], **SOFT_LIKELIHOOD_MATRIX['stable']},
    'time_bomb': {**FINANCIAL_LIKELIHOOD_MATRIX['time_bomb'], **SOFT_LIKELIHOOD_MATRIX['time_bomb']}
}

# Underlying growth parameters
GROWTH_MU = np.array([0.097, 0.01, -0.048])
GROWTH_SCALE = np.array([0.1, 0.01, 0.05])


def get_soft_indicator_xarray() -> xr.Dataset:
    """
    Convert the SOFT_LIKELIHOOD_MATRIX into an xarray dataset
    """
    # First we need to add yellow to the values
    matrix_copy = SOFT_LIKELIHOOD_MATRIX.copy()
    for category in matrix_copy:
        for indicator in matrix_copy[category]:
            matrix_copy[category][indicator]['yellow'] = 1.0 - sum(
                matrix_copy[category][indicator].values()
            )

    likelihoods = xr.Dataset(
        {
            indicator: (
                ['category', 'color'],
                [[matrix_copy[category][indicator][color] for color in ['red', 'yellow', 'green']]
                    for category in ['high_growth', 'stable', 'time_bomb']]
            )
            for indicator in matrix_copy['high_growth']
        },
        coords={
            'category': ['high_growth', 'stable', 'time_bomb'],
            'color': ['red', 'yellow', 'green']
        }
    )
    return likelihoods


def synthetic_indicator_cost(mu: float) -> float:
    """
    Add noise to an indicator cost

    :param mu:  Expected budget cost
    """
    return max(0.5, np.random.normal(mu, mu / 10))


def synthetic_extractor_noise(distribution: np.ndarray) -> np.ndarray:
    """
    Simulate soft-indicator noise incurred by the soft-indicator extractor, using the color confusion matrix.

    :param distribution:  [..., 3] array of color probabilities
    :return:              [..., 3] array with applied noise
    """
    color_confusion = np.array(
        [
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
        ]
    )
    return np.dot(distribution, color_confusion)


def sample_independent_indicators(specified_colors=None):
    """
    Generate random values for independent indicators based on specified interval colors,
    defaulting to yellow for unspecified indicators.

    Args:
        specified_colors (dict): Dictionary mapping indicator names to colors ('red', 'yellow', 'green')

    Returns:
        dict: Dictionary with random values for independent indicators only
    """
    if specified_colors is None:
        specified_colors = {}

    # Only sample independent indicators
    result = {}
    for indicator, intervals in INDEPENDENT_INDICATOR_INTERVALS.items():
        color = specified_colors.get(indicator, 'yellow')
        lower, upper = intervals[color]

        # Sample based on indicator type
        if indicator in ['revenue_growth', 'ebitda_growth']:
            # Growth rates: beta distribution for more realistic values
            # Transform beta distribution to match our interval
            alpha = 2
            beta = 3 if color == 'red' else (2 if color == 'yellow' else 1.5)
            value = (upper - lower) * np.random.beta(alpha, beta) + lower

        elif indicator in ['working_capital_ratio', 'leverage']:
            # Ratios: log-normal for positive skew
            mu = np.log((lower + upper) / 2)
            sigma = 0.4  # Reduced from 0.5 for less extreme values
            value = min(upper, max(lower, np.random.lognormal(mu, sigma)))

        else:
            # Other metrics: uniform distribution
            value = np.random.uniform(lower, upper)

        result[indicator] = round(value, 3)

    return result


def calculate_derived_indicators(independent_indicators):
    """
    Calculate derived indicators from independent ones.

    Args:
        independent_indicators: Dictionary of independent indicators

    Returns:
        Dictionary containing all indicators (independent and derived)
    """
    # Constants
    TAX_RATE = 0.25
    COST_OF_DEBT = 0.05
    COST_OF_EQUITY = 0.10

    # Start with copy of independent indicators
    all_indicators = independent_indicators.copy()

    # 1. Calculate WACC
    leverage = all_indicators['leverage']
    wacc = (COST_OF_EQUITY * 1 / (1 + leverage) +
            COST_OF_DEBT * (1 - TAX_RATE) * leverage / (1 + leverage))

    # 2. Calculate interest burden based on leverage
    interest_burden = leverage * COST_OF_DEBT / (1 + leverage)

    # 3. Calculate net profit margin from operating margin
    all_indicators['net_profit_margin'] = (
            all_indicators['operating_margin'] *
            (1 - TAX_RATE) *
            (1 - interest_burden)
    )

    # 4. Calculate ROIC from operating margin and asset turnover
    all_indicators['roic'] = (
            all_indicators['operating_margin'] *
            all_indicators['asset_turnover'] *
            (1 - TAX_RATE)
    )

    # Use small positive number if ROIC is zero to avoid division issues
    roic_for_calc = all_indicators['roic'] if all_indicators['roic'] != 0 else 1e-6

    # Calculate ROE from ROIC and leverage
    all_indicators['roe'] = (
            roic_for_calc * (1 + leverage) -
            (COST_OF_DEBT * leverage * (1 - TAX_RATE))
    )

    # Calculate CFROI based on ROIC and reinvestment
    all_indicators['cfroi'] = roic_for_calc * (1 - all_indicators['reinvestment_rate'])

    # Calculate valuation metrics using growth and returns
    growth_factor = (all_indicators['revenue_growth'] + all_indicators['ebitda_growth']) / 2
    growth_for_calc = growth_factor if growth_factor < wacc else wacc - 1e-6

    # PE ratio using simplified Gordon growth
    all_indicators['pe_ratio'] = (1 + growth_for_calc) / (wacc - growth_for_calc)

    # EV/EBITDA roughly tracks with PE but typically lower
    all_indicators['ev_ebitda'] = all_indicators['pe_ratio'] * 0.8

    # P/B ratio follows from ROE and PE
    all_indicators['pb_ratio'] = all_indicators['pe_ratio'] * all_indicators['roe']

    # 8. Calculate FCF growth based on EBITDA growth and reinvestment
    all_indicators['fcf_growth'] = (
            all_indicators['ebitda_growth'] *
            (1 - 0.1)  # Assume 10% change in reinvestment rate
    )

    return all_indicators


def generate_financials(independent_indicators):
    """
    Generate synthetic financial statements that are consistent with given financial indicators.
    Uses revenue as the anchor and derives other values to maintain ratio consistency.

    Args:
        independent_indicators (dict): Dictionary of financial indicators

    Returns:
        dict: Dictionary containing P&L, balance sheet, and market data
    """
    # Constants
    TAX_RATE = 0.25

    # First calculate all derived indicators
    indicators = calculate_derived_indicators(independent_indicators)

    # STEP 1: Revenue and Income Statement
    # This is our base scale and determines all absolute values
    base_revenue = np.random.uniform(100e6, 10e9)

    income_statement = {
        'revenue': base_revenue,
        'gross_profit': base_revenue * indicators['gross_margin'],
        'operating_income': base_revenue * indicators['operating_margin'],
        'net_income': base_revenue * indicators['net_profit_margin']
    }

    # Add EBITDA components
    da_rate = np.random.uniform(0.2, 0.4)
    income_statement['depreciation_amortization'] = income_statement['operating_income'] * da_rate
    income_statement['ebitda'] = income_statement['operating_income'] + income_statement['depreciation_amortization']

    # STEP 2: Capital Structure and Balance Sheet
    # Start with equity from ROE (this will ensure ROE is exact)
    balance_sheet = {}
    balance_sheet['total_equity'] = income_statement['net_income'] / indicators['roe']

    # Debt from leverage target (this ensures leverage is exact)
    balance_sheet['total_debt'] = balance_sheet['total_equity'] * indicators['leverage']

    # Total assets from asset turnover (this ensures asset turnover is exact)
    balance_sheet['total_assets'] = income_statement['revenue'] / indicators['asset_turnover']

    # STEP 3: Operating Capital Components
    # Calculate NOPAT and invested capital from ROIC
    nopat = income_statement['operating_income'] * (1 - TAX_RATE)
    roic_for_calc = indicators['roic'] if indicators['roic'] != 0 else 1e-6
    invested_capital = nopat / roic_for_calc

    # Split invested capital into working capital and fixed capital
    working_capital_percent = 0.3  # Assume 30% of invested capital is working capital
    target_working_capital = invested_capital * working_capital_percent

    # Handle working capital ratio calculations more safely
    # If ratio is R, then:
    # current_assets = R * current_liabilities
    # working_capital = current_assets - current_liabilities = target_working_capital
    # Therefore: target_wc = R*CL - CL = CL*(R-1)
    # So: CL = target_wc/(R-1) but we need to handle R ≈ 1 carefully

    wcr = indicators['working_capital_ratio']
    if abs(wcr - 1.0) < 0.01:  # If ratio is very close to 1
        # Use a reasonable default relationship
        balance_sheet['current_liabilities'] = target_working_capital
        balance_sheet['current_assets'] = target_working_capital * wcr
    else:
        balance_sheet['current_liabilities'] = abs(target_working_capital / (wcr - 1))
        balance_sheet['current_assets'] = balance_sheet['current_liabilities'] * wcr

    balance_sheet['working_capital'] = balance_sheet['current_assets'] - balance_sheet['current_liabilities']

    # Fixed assets (balancing item for invested capital)
    balance_sheet['fixed_assets'] = invested_capital - balance_sheet['working_capital']

    # STEP 4: Cash Flows
    # Use CFROI to determine operating cash flow (this ensures CFROI is exact)
    cash_flow = {
        'operating_cash_flow': invested_capital * indicators['cfroi'],
        # Use reinvestment rate to determine capex (this ensures reinvestment rate is exact)
        'capex': -income_statement['net_income'] * indicators['reinvestment_rate']
    }
    cash_flow['free_cash_flow'] = cash_flow['operating_cash_flow'] + cash_flow['capex']

    # STEP 5: Market Values
    # Start with P/E to get market cap (this ensures P/E is exact)
    market_data = {
        'market_cap': income_statement['net_income'] * indicators['pe_ratio'],
        # Use EV/EBITDA to get enterprise value (this ensures EV/EBITDA is exact)
        'enterprise_value': income_statement['ebitda'] * indicators['ev_ebitda'],
        'book_value': balance_sheet['total_equity'],
        'shares_outstanding': np.random.uniform(50e6, 1e9)
    }
    market_data['price_per_share'] = market_data['market_cap'] / market_data['shares_outstanding']

    # STEP 6: Previous Period Values for Growth Metrics
    prev_period = {
        'revenue': base_revenue / (1 + indicators['revenue_growth']),
        'fcf': cash_flow['free_cash_flow'] / (1 + indicators['fcf_growth']),
        'ebitda': income_statement['ebitda'] / (1 + indicators['ebitda_growth'])
    }

    # Verification calculations
    verification = {
        'roic_calc': nopat / (invested_capital if invested_capital != 0 else 1e-6),
        'operating_margin_calc': income_statement['operating_income'] / (
            income_statement['revenue'] if income_statement['revenue'] != 0 else 1e-6),
        'gross_margin_calc': income_statement['gross_profit'] / (
            income_statement['revenue'] if income_statement['revenue'] != 0 else 1e-6),
        'net_profit_margin_calc': income_statement['net_income'] / (
            income_statement['revenue'] if income_statement['revenue'] != 0 else 1e-6),
        'leverage_calc': balance_sheet['total_debt'] / (
            balance_sheet['total_equity'] if balance_sheet['total_equity'] != 0 else 1e-6),
        'working_capital_ratio_calc': balance_sheet['current_assets'] / (
            balance_sheet['current_liabilities'] if balance_sheet['current_liabilities'] != 0 else 1e-6),
        'asset_turnover_calc': income_statement['revenue'] / (
            balance_sheet['total_assets'] if balance_sheet['total_assets'] != 0 else 1e-6),
        'revenue_growth_calc': (income_statement['revenue'] / (
            prev_period['revenue'] if prev_period['revenue'] != 0 else 1e-6)) - 1,
        'fcf_growth_calc': (cash_flow['free_cash_flow'] / (
            prev_period['fcf'] if prev_period['fcf'] != 0 else 1e-6)) - 1,
        'ebitda_growth_calc': (income_statement['ebitda'] / (
            prev_period['ebitda'] if prev_period['ebitda'] != 0 else 1e-6)) - 1,
        'pe_ratio_calc': market_data['market_cap'] / (
            income_statement['net_income'] if income_statement['net_income'] != 0 else 1e-6),
        'ev_ebitda_calc': market_data['enterprise_value'] / (
            income_statement['ebitda'] if income_statement['ebitda'] != 0 else 1e-6),
        'pb_ratio_calc': market_data['market_cap'] / (
            balance_sheet['total_equity'] if balance_sheet['total_equity'] != 0 else 1e-6),
        'reinvestment_rate_calc': -cash_flow['capex'] / (
            income_statement['net_income'] if income_statement['net_income'] != 0 else 1e-6),
        'roe_calc': income_statement['net_income'] / (
            balance_sheet['total_equity'] if balance_sheet['total_equity'] != 0 else 1e-6),
        'cfroi_calc': cash_flow['operating_cash_flow'] / (invested_capital if invested_capital != 0 else 1e-6)
    }

    # Calculate relative errors
    errors = {}
    for calc_metric in verification.keys():
        if calc_metric.endswith('_calc'):
            base_metric = calc_metric[:-5]  # Remove '_calc' suffix
            if base_metric in indicators:
                calc_value = verification[calc_metric]
                target_value = indicators[base_metric]
                if abs(target_value) < 1e-6:
                    # For values very close to zero, use absolute error instead of relative
                    errors[base_metric] = abs(calc_value - target_value)
                else:
                    errors[base_metric] = abs((calc_value - target_value) / target_value)

    # Add error analysis to verification
    verification['error_analysis'] = errors

    # Check tolerance violations
    tolerance_thresholds = DEFAULT_TOLERANCES

    tolerance_violations = {
        metric: {
            'error': error,
            'threshold': tolerance_thresholds[metric],
            'status': 'EXCEEDED' if error > tolerance_thresholds[metric] else 'OK'
        }
        for metric, error in errors.items()
        if metric in tolerance_thresholds
    }

    verification['tolerance_check'] = tolerance_violations

    # Calculate quality score
    metrics_within_tolerance = sum(1 for v in tolerance_violations.values() if v['status'] == 'OK')
    total_metrics = len(tolerance_violations)
    verification['quality_score'] = metrics_within_tolerance / total_metrics if total_metrics > 0 else 0

    financials = {
        'income_statement': income_statement,
        'balance_sheet': balance_sheet,
        'cash_flow': cash_flow,
        'market_data': market_data,
        'prev_period': prev_period,
        'verification': verification
    }

    # Warning for low quality score
    if verification['quality_score'] < 0.8:
        print(f"Warning: Generated financials have low quality score: {verification['quality_score']:.1%}")
        print("Metrics exceeding tolerance:")
        for metric, check in tolerance_violations.items():
            if check['status'] == 'EXCEEDED':
                print(f"  {metric}: error {check['error']:.1%} vs threshold {check['threshold']:.1%}")

    return financials


def generate_synthetic_stocks(n_stocks: int, as_dataset: bool = True) -> List[Dict[str, Any]] | xr.Dataset:
    """
    Generate a specified number of synthetic stocks with full financial data.

    Args:
        n_stocks: Number of synthetic stocks to generate
        as_dataset: Whether to return the stocks as an xarray Dataset

    Returns:
        List of dictionaries containing company category, indicators, and financials
        or xarray.Dataset
    """
    # Company category probabilities (can be adjusted)
    category_probs = {
        'high_growth': 0.3,  # 30% are high growth
        'stable': 0.5,  # 50% are stable
        'time_bomb': 0.2  # 20% are time bombs
    }

    stocks = []

    for _ in range(n_stocks):
        # Step 1: Sample company category
        category = np.random.choice(
            list(category_probs.keys()),
            p=list(category_probs.values())
        )

        # Step 2: Sample indicator colors based on likelihood matrix

        # Combine financial and soft indicators for this category
        category_likelihoods = {
            **FINANCIAL_LIKELIHOOD_MATRIX[category],
            **SOFT_LIKELIHOOD_MATRIX[category]
        }

        indicator_colors = {}

        for indicator, probs in category_likelihoods.items():
            # For each indicator, calculate probabilities for red, yellow, green
            red_prob = probs['red']
            green_prob = probs['green']
            yellow_prob = 1 - (red_prob + green_prob)

            # Sample color based on probabilities
            color = np.random.choice(
                ['red', 'yellow', 'green'],
                p=[red_prob, yellow_prob, green_prob]
            )

            indicator_colors[indicator] = color

        # Step 3: Sample actual indicator values based on colors
        indicators = sample_independent_indicators(indicator_colors)

        # Step 4: Generate financial statements
        financials = generate_financials(indicators)

        # Step 5: Package everything together
        stock_data = {
            'category': category,
            'sampled_colors': indicator_colors,
            'indicators': indicators,
            'financials': financials
        }

        stocks.append(stock_data)

    if as_dataset:
        return stocks_to_xarray(stocks)

    return stocks


def print_stock_summary(stock_data: Dict[str, Any]) -> None:
    """
    Print a summary of a generated stock's key metrics.

    Args:
        stock_data: Dictionary containing stock information
    """
    print(f"\nCompany Category: {stock_data['category']}")

    print("\nKey Indicators:")
    for indicator, value in stock_data['indicators'].items():
        color = stock_data['sampled_colors'][indicator]
        if indicator in INDEPENDENT_INDICATOR_INTERVALS:  # Only show independent indicators
            print(f"{indicator}: {value:.3f} ({color})")

    fin = stock_data['financials']
    print("\nKey Financials (in millions):")
    print(f"Revenue: ${fin['income_statement']['revenue'] / 1e6:.1f}M")
    print(f"Net Income: ${fin['income_statement']['net_income'] / 1e6:.1f}M")
    print(f"Total Assets: ${fin['balance_sheet']['total_assets'] / 1e6:.1f}M")
    print(f"Market Cap: ${fin['market_data']['market_cap'] / 1e6:.1f}M")

    print("\nQuality Score:", f"{fin['verification']['quality_score']:.1%}")


def stocks_to_xarray(stocks) -> xr.Dataset:
    """
    Convert list of synthetic stocks into an xarray Dataset with both financial and soft indicators.
    """
    # Get list of all indicators
    financial_indicators = list(INDEPENDENT_INDICATOR_INTERVALS.keys())
    soft_indicators = ['disruptor_stagnant',
        'incumbent_stagnant',
        'turnaround_plan',
        'succession_risk',
        'mismanagement',
        'geopolitical_risk',
        'environmental_risk',
        'product_moat',
        'supply_chain_resilience',
        'regulatory_pressure',
        'innovation_pipeline',
        'customer_concentration',
        'brand_strength',
        'digital_transformation']
    all_indicators = financial_indicators + soft_indicators
    n_stocks = len(stocks)

    # Create empty arrays
    values = np.full((n_stocks, len(all_indicators)), np.nan)  # NaN for soft indicators
    colors = np.empty((n_stocks, len(all_indicators)), dtype='U6')  # U6 for 'yellow'

    # Fill arrays
    for i, stock in enumerate(stocks):
        for j, indicator in enumerate(all_indicators):
            color = stock['sampled_colors'][indicator]
            colors[i, j] = color
            if indicator in financial_indicators:
                values[i, j] = stock['indicators'][indicator]
            # else:
            #     # Soft indicators, bernoulli sample based on color
            #     if color == 'red':
            #         values[i, j] = np.random.binomial(1, 0.1)  # 10% change for red
            #     elif color == 'green':
            #         values[i, j] = np.random.binomial(1, 0.9)
            #     else:
            #         values[i, j] = np.random.binomial(1, 0.5)  # yellow

    # Create asset coordinates
    asset_coords = [f'asset_{i}' for i in range(n_stocks)]

    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            'val': (('asset', 'indicator'), values),
            'ryg': (('asset', 'indicator'), colors)
        },
        coords={
            'asset': asset_coords,
            'indicator': all_indicators,
            'indicator_type': ('indicator', ['financial' if ind in financial_indicators else 'soft'
                for ind in all_indicators])
        }
    )

    # Add stock categories as coordinate
    ds = ds.assign_coords(
        category=('asset', [stock['category'] for stock in stocks])
    )

    # Add some 2025 simulated growths to ds
    company_types = ['high_growth', 'stable', 'time_bomb']
    company_type = np.array([company_types.index(val) for val in ds.category.values])
    growth_2025 = rng.normal(loc=GROWTH_MU[company_type], scale=GROWTH_SCALE[company_type])
    ds['growth_2025'] = xr.DataArray(growth_2025, dims='asset')

    # Add attributes for metadata
    ds.attrs['description'] = 'Synthetic stock data with financial and soft indicators'
    ds.val.attrs['units'] = 'ratio'
    ds.val.attrs['description'] = 'Numerical values (financial indicators only)'
    ds.ryg.attrs['description'] = 'Red/Yellow/Green classification of indicators'

    return ds
