"""
CRISTAL Models
"""
import copy
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Union, Any, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from scipy import stats
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cristal import stats as cristal_stats, synthetic as synth
from cristal.env import PROJECT_ROOT, REPORTS_DIR
from cristal.llm import AVAILABLE_MODELS
from cristal.llm.base import LLMBaseClass, SoftIndicatorExtractionError
from cristal.llm.codegen.model import LLMCodeGen
from cristal.llm.codegen.validator import LLMCodeGenException, LLMCodeGenValidator, load_code
from cristal.llm.wrangling import read_reports
from cristal.logger import logger
from cristal.utils import BudgetMode, describe_class

Indicators = synth.Indicators


class CRISTAL:
    """
    Coherent Reliable Intentional Synthesis of Truthful Analysis Logic.

    Through statistical modeling of asset/company properties (i.e. their asset class, and growth),
    CRISTAL refines its predictive abilities on these properties through active inference on
    observed asset attributes, namely their 'indicators'.

    Equipped with the ground-truth parameters that make up the growth distribution (mu_g, sigma_g), CRISTAL can
    make an informed estimate of an asset's upcoming growth by its categorical likelihood (mixture weights).
    The growth distribution follows a Gaussian Mixture Model, formed by the three asset classes:
    High Growth, Stable, or Time-Bomb.

    Attributes:
        n_assets (int): Number of assets in the dataset.
        classes (Dict[str, str]): Mapping of class codes to their titles.
        class_idx (Dict[str, int]): Mapping of class codes to their indices.
        n_classes (int): Number of classes.
        mu_g (numpy.ndarray): Mean growth parameters.
        sigma_g (numpy.ndarray): Standard deviation of growth parameters.
        quantiles (List[float]): List of quantiles for growth predictions.
        asset_memory_depth (int): Depth of memory in steps for inference model updates.
        budget (float): Remaining budget for performing asset evaluations.
        budget_mode (BudgetMode): What budget mode we are using.
        llm (LLMBaseClass): Loaded LLM at CRISTAL's disposal for various tasks.
        stocks (xarray.Dataset): Dataset for the assets/stocks. Also where CRISTAL's belief model parameters are stored.
        indicators (xarray.Dataset): Indicator dataset of likelihoods of a color given class.
        use_codegen (bool): Use our codegen module instead of the hardcoded functions in stats
        codegen_llm (LLMCodeGen): Optional LLMCodeGen instance if using codegen
        _cached_reports (Optional[Dict[str, str]]): Cached reports.
        _version (str): Version.
    """

    # Class-level annotations for introspection
    n_assets: int
    classes: Dict[str, str]
    class_idx: Dict[str, int]
    n_classes: int
    mu_g: np.ndarray
    sigma_g: np.ndarray
    quantiles: List[float]
    asset_memory_depth: int
    budget: float
    budget_mode: BudgetMode
    llm: LLMBaseClass
    stocks: xr.Dataset
    indicators: xr.Dataset
    use_codegen: bool
    codegen_llm: Optional[LLMCodeGen]
    _cached_reports: Optional[Dict[str, str]]
    _version: str = "0.2"



    def __init__(
        self,
        stocks_dataset: xr.Dataset,
        llm: LLMBaseClass,
        asset_memory_depth: int = 20,
        budget: float = 100,
        budget_mode: BudgetMode = BudgetMode.WALL,
        use_codegen: bool = False,
    ):
        self.alpha_posterior = None
        self.beta_posterior = None
        """
        Copy the stocks Dataset, and add new dimensions for CRISTAL's class prediction, and quantile predictions

        :param stocks_dataset:      xr.Dataset from synth.generate_synthetic_stocks
        :param llm:                 Initialized LLM model
        :param asset_memory_depth:  How many inference updates to remember for each asset
        :param budget:              Budget for CRISTAL to spend on inference
        :param budget_mode:         Budget mode
        :param use_codegen:         Use the codegen module instead of hardcoded functions from our stats lib
        """
        # Copy the stocks Dataset and add new dimension for CRISTAL's class prediction, and quantile predictions
        self.stocks: xr.Dataset = stocks_dataset.copy()
        # Contains e.g. the financials
        self.stocks_json: Dict[str, synth.AssetJSON] = synth.load_stocks(as_dataset=False)
        self.asset_memory_depth = asset_memory_depth
        self.budget = budget
        self.budget_mode = budget_mode
        assert budget_mode in BudgetMode, f"Invalid budget mode {budget_mode}"
        self.llm = llm

        # Codegen
        self.use_codegen = use_codegen
        self.codegen_llm = None
        # Track loaded codegen functions with this
        # Call the run_codegen_suite method to populate this lookup
        self.loaded_codegens = {}
        if use_codegen:
            # Init and assign LLMCodeGen instance
            self.codegen_llm = LLMCodeGen(self.llm)

        self.n_assets = len(self.stocks.asset)
        self.classes = {
            "high_growth": "High Growth",
            "stable": "Stable",
            "time_bomb": "Time Bomb"
        }
        self.class_idx = {
            "high_growth": 0,
            "stable": 1,
            "time_bomb": 2
        }
        self.n_classes = len(self.classes)

        # Known growth params
        self.mu_g = synth.GROWTH_MU
        self.sigma_g = synth.GROWTH_SCALE
        # Shorthand
        self.quantiles = [0.025, 0.16, 0.5, 0.84, 0.975]
        # Initialize indicators dataset
        self.indicators: xr.Dataset = synth.get_indicator_dataset()

        # Initialize priors
        self.initialize_priors()
        self.initialize_posteriors()
        # Also initializes a couple attributes
        self.reset()

    def to_json(self):
        """
        Return certain model attributes as JSON.
        Naturally, we don't want to serialize the datasets etc.
        """
        return {
            "version": self._version,
            "indicators": {
                "indicator": self.indicators.indicator.values.tolist(),
                "qualitative": self.indicators.qualitative.values.tolist(),
                "utility": self.indicators.utility.values.tolist(),
                "observations": self.indicators.observations.values.tolist(),
                "ranking": self.indicators.ranking.values.tolist(),
                "cost": self.indicators.cost.values.tolist(),
            },
            "llm": self.llm.to_json(),
            "asset_memory_depth": self.asset_memory_depth,
            "ground_truth": {
                "classes": self.classes,
                "class_idx": self.class_idx,
                "mu_g": self.mu_g.tolist(),
                "sigma_g": self.sigma_g.tolist(),
            },
            "quantiles": self.quantiles,
        }

    def reset(self):
        """
        Reset the model to its initial state
        """
        # Quantile predictions -- Quantiles initialized based on uniform class belief
        init_params = [(m, sig) for m, sig in zip(self.mu_g, self.sigma_g)]
        mix_weights = np.full(self.n_classes, 1 / self.n_classes)
        quantile_values = cristal_stats.MixtureDistribution.compute_quantiles(
            distributions=init_params,
            weights=mix_weights,
            quantiles=self.quantiles
        )
        # Broadcast across assets and memory depth
        init_quantiles = np.broadcast_to(
            quantile_values.reshape(1, len(self.quantiles), 1),
            (self.n_assets, len(self.quantiles), self.asset_memory_depth)
        )
        self.stocks['quantile_pred'] = xr.DataArray(
            init_quantiles.copy(),
            dims=['asset', 'quantile', 'time'],
            coords={
                'quantile': self.quantiles,
                'time': list(range(self.asset_memory_depth))
            }
        )
        # Asset priors
        # We will use a Dirichlet prior for our Categorical variable
        init_alpha_prior = np.ones((self.n_assets, self.n_classes, self.asset_memory_depth))  # No bias
        self.stocks['alphas'] = xr.DataArray(
            init_alpha_prior,
            dims=['asset', 'asset_class', 'time'],
            coords={
                'asset_class': list(self.classes.keys()),
                'time': list(range(self.asset_memory_depth))
            },
        )
        self.stocks['prior'] = xr.DataArray(
            init_alpha_prior.copy() / self.n_classes,
            dims=['asset', 'asset_class', 'time'],
            coords={
                'asset_class': list(self.classes.keys()),
                'time': list(range(self.asset_memory_depth))
            },
        )

        # Indicator Dataset
        ### TODO: initialize to uniform prior IFF exp2
        self.indicators: xr.Dataset = synth.get_indicator_dataset()
        self._cached_reports = {}

    def ind_likelihood(self, name: str) -> np.ndarray:
        """
        Helper function
        Return the likelihoods for the given indicator
        """
        return self.indicators.likelihood.sel(indicator=name).values

    @property
    def ranking_idx(self) -> np.ndarray:
        """Ranking indices based on utility"""
        return self.indicators.utility.argsort()[::-1].values

    @property
    def sp_ranking_idx(self) -> np.ndarray:
        """
        The special ranking index ranks by utility, but has a higher order ranking
        that places quantitative indicators first, then qualitative
        """
        quant_masked = self.indicators.where(~self.indicators.qualitative)
        quant_sorted = quant_masked.utility.argsort()[::-1].values
        # Only grab the last few that are quants
        quant_sorted = quant_sorted[-len(Indicators.quantitative):]
        # Similar for qualitative
        qual_masked = self.indicators.where(self.indicators.qualitative)
        qual_sorted = qual_masked.utility.argsort()[::-1].values
        qual_sorted = qual_sorted[-len(Indicators.qualitative):]
        # Combine
        return np.concatenate([quant_sorted, qual_sorted])

    def _update_quantile_pred(self, asset: xr.Dataset, quantile_pred):
        """
        Update the quantile history

        :param asset: xr.Dataset for the selected asset
        """
        # Update the quantile predictions -- place the new value at the end
        self.stocks['quantile_pred'].loc[asset.asset] = np.roll(
            self.stocks['quantile_pred'].loc[asset.asset], -1, axis=1
        )
        self.stocks['quantile_pred'].loc[asset.asset][:, -1] = quantile_pred

    def _update_alphas(self, asset: xr.Dataset, alphas):
        """
        Update alphas

        :param asset: xr.Dataset for the selected asset
        """
        # Update the alphas -- place the new value at the end
        self.stocks['alphas'].loc[asset.asset] = np.roll(
            self.stocks['alphas'].loc[asset.asset], -1, axis=1
        )
        self.stocks['alphas'].loc[asset.asset][:, -1] = alphas

    def _update_prior(self, asset: xr.Dataset, posterior):
        """
        Update prior from provided posterior

        :param asset: xr.Dataset for the selected asset
        """
        # Update the prior -- place the new value at the end
        self.stocks['prior'].loc[asset.asset] = np.roll(
            self.stocks['prior'].loc[asset.asset], -1, axis=1
        )
        self.stocks['prior'].loc[asset.asset][:, -1] = posterior

    def _update_likelihoods(self, indicator_name: str, color_pred: np.ndarray):
        """
        Update the likelihoods for the given indicator based on the observed color prediction

        :param indicator_name: Name of the indicator
        :param color_pred: Observed color prediction as a one-hot encoded array
        """
        # TODO: to be implemented
        # @michael did you want a shot at that? i cant get my head around it for some reason
        # self.indicators.likelihood = self.ind_likelihood(indicator_name) + color_pred

    def run_codegen_suite(self, validators: list[Type[LLMCodeGenValidator]]) -> None:
        """
        Query the LLM to load all the codegen functions based on the provided validators

        :param validators:  Codegen functions to generate
        :return:            None
        """
        for val in validators:
            logger.info(f"Loading codegen function '{val.function_name}'...")
            if val.function_name in self.loaded_codegens:
                logger.warning(f"Already loaded codegen function '{val.function_name}'. Continuing")
                continue

            try:
                result = self.codegen_llm.generate_function(
                    val, max_retries=5, budget=60
                )
            except LLMCodeGenException as e:
                # Raise this again
                raise e
            else:
                # Add to our lookup
                self.loaded_codegens[val.function_name] = result

        logger.info(f"Finished Codegen suite")

    def codegen_pipe(self, function_name: str, function_kwargs: dict) -> Any:
        """
        Hacky interim solution to pipe a potential codegen function to the
        """
        if not self.loaded_codegens:
            raise RuntimeError("Codegen function requested but it has not been initialized.")

        if function_name not in self.loaded_codegens:
            raise ValueError(f"Function '{function_name}' not in loaded codegen functions")

        if self.use_codegen:
            return self.loaded_codegens[function_name](**function_kwargs)
        else:
            if function_name == "dirichlet_bayes":
                return cristal_stats.dirichlet_bayes(**function_kwargs)
            else:
                raise ValueError(f"Undefined method '{function_name}' passed into codegen_pipe")

    def initialize_priors(self):
        """
        Initialize the alpha and beta priors to be uniform.
        """
        # Uniform alpha prior: equal probability for each class
        self.alpha_prior = np.ones(self.n_classes)

        # Uniform beta prior: equal likelihood for each color across all indicators and classes
        num_indicators = self.indicators.indicator.size
        num_colors = self.indicators.likelihood.shape[2]
        self.beta_prior = np.ones((self.n_classes, num_indicators, num_colors))

    def initialize_posteriors(self):
        """
        Initialize the alpha and beta posteriors to be uniform.
        """
        # Uniform alpha posterior: equal probability for each class
        self.alpha_posterior = np.ones(self.n_classes) / self.n_classes

        # Uniform beta posterior: equal likelihood for each color across all indicators and classes
        num_indicators = self.indicators.indicator.size
        num_colors = self.indicators.likelihood.shape[2]
        self.beta_posterior = np.ones((self.n_classes, num_indicators, num_colors)) / num_colors

    def likelihood_inference(self, data: dict):
        """
        Updates the likelihood matrix based on the observed data.

        Args:
            data: A dictionary containing the observed data for each asset. The keys are the asset names,
                and the values are dictionaries with 'category' and 'indicators' keys. The 'category' key contains
                the class label of the asset, and the 'sampled_colors' key contains a dictionary of the observed
                indicators their their true colors.
        """
        if not hasattr(self, "alpha_prior") or not hasattr(self, "beta_prior"):
            raise AttributeError("Required attributes 'alpha_prior' or 'beta_prior' are not defined.")

        # Reshape the dictionary input data to a NumPy array
        # N: number of observations, J: number of indicators
        indicators = list(data.values())[0]['sampled_colors'].keys()
        N = len(data)
        J = len(indicators)
        X = np.zeros((N, J), dtype=int)

        for i, (asset_name, asset_data) in enumerate(data.items()):
            for j, indicator in enumerate(indicators):
                x_ij = asset_data['sampled_colors'][indicator]
                x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
                X[i, j] = x_ij

        # Update beta (likelihood matrix).
        for j in range(J):  # Iterate over indicators
            for k_val, category in enumerate(self.classes.keys()):  # Iterate over classes
                for c in range(self.beta_prior.shape[2]):  # Iterate over colors
                    self.beta_posterior[k_val, j, c] = self.beta_prior[k_val, j, c] + np.sum(
                        (X[:, j] == c) & (np.array([asset['category'] for asset in data.values()]) == category)
                    )
        epsilon = 1e-20  # Small value to prevent division by zero
        self.indicators['likelihood'] = (('class', 'indicator', 'color'),
        self.beta_posterior / (np.sum(self.beta_posterior, axis=2, keepdims=True) + epsilon))

    def add_budget(self, budget: float):
        """
        Add budget to the model
        """
        self.budget += budget

    def get_report(self, asset_id: str) -> str:
        """
        Return the cached report for the selected asset
        """
        if asset_id in self._cached_reports:
            return self._cached_reports[asset_id]

        report_text = read_reports(
            report_dir=str(os.path.join(REPORTS_DIR, asset_id)),
        )
        size_limit = 1e7  # in bytes
        self._cached_reports[asset_id] = report_text
        if sys.getsizeof(self._cached_reports) > size_limit:
            # Clear cache
            self._cached_reports = {}

        return report_text

    def rank_indicators_deprecated(self, prior, candidates) -> list:
        """
        Rank indicators by expected utility using expected log Bayes factor for growth prediction

        Also updates observed indicator utility average
        """
        # For tracking utility
        indicator_tuples = []

        for can in candidates:
            utility = 0

            class_color_matrix = self.ind_likelihood(can['name'])  # [classes, colors]
            # Sum over the colors for each asset-class
            color_likelihoods = prior.reshape(1, -1) @ class_color_matrix  # [p(r|p), p(y|p), p(g|p)]
            color_likelihoods = color_likelihoods.reshape(-1, )
            logger.debug(f"Current likelihood matrix: {self.indicators.likelihood.values}")
            # TODO -- optimize
            for ic, c in enumerate(["red", "yellow", "green"]):
                # P(c | p_k)
                p_col = color_likelihoods[ic]

                if p_col > 0:
                    # Posterior class probabilities if we observe color
                    cls_likelihood = class_color_matrix[:, ic]
                    gamma = prior * cls_likelihood
                    if np.sum(gamma) > 1e-6:
                        gamma = gamma / np.sum(gamma)

                        # Expected log Bayes factor is expected log ratio of the densities
                        # Note: we need to integrate over possible g values
                        g_points = np.linspace(
                            min(self.mu_g - 3 * max(self.sigma_g)),
                            max(self.mu_g + 3 * max(self.sigma_g)),
                            100
                        )

                        for g in g_points:
                            # P(g|posterior)
                            post_prob = sum(
                                gamma[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                    for k in range(self.n_classes)
                            )
                            # P(g|prior)
                            prior_prob = sum(
                                prior[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                    for k in range(self.n_classes)
                            )

                            if post_prob > 0 and prior_prob > 0:
                                # Weight by posterior probability of g
                                utility += p_col * post_prob * (np.log(post_prob) - np.log(prior_prob)) * \
                                           (g_points[1] - g_points[0])

            # Add to list
            indicator_tuples.append((can, utility))

            # Update the utility of this indicator
            obs = self.indicators.observations.sel(indicator=can['name']).item()
            if obs == 0:
                # First observation
                self.indicators.observations.loc[can['name']] = 1
                self.indicators.utility.loc[can['name']] = utility
            else:
                # Running average
                current_avg = self.indicators.utility.loc[can['name']].item()
                self.indicators.observations.loc[can['name']] = obs + 1
                self.indicators.utility.loc[can['name']] += ((utility - current_avg) / obs)
                # self.indicator_utility_avg[j, 1] += ((utility - self.indicator_utility_avg[j, 1]) /
                #                                      self.indicator_utility_avg[j, 0])

        # Return the sorted list of candidates, descending
        return [tup[0] for tup in sorted(indicator_tuples, key=lambda x: x[1], reverse=True)]

    def rank_indicators(self, prior, candidates) -> list:
        """
        Rank indicators by expected utility using expected log Bayes factor for growth prediction

        Also updates observed indicator utility average
        """
        # For tracking utility
        indicator_tuples = []

        for can in candidates:
            utility = 0
            ### TODO: does this give CRISTAL the likelihoods we dont want it to have in exp2 anymore?
            ### maybe iniialize a uniform prior for the likelihoods?
            ### update belief in likelihoods based on the data
            class_color_matrix = self.ind_likelihood(can['name'])  # [classes, colors]
            # Sum over the colors for each asset-class
            color_likelihoods = prior.reshape(1, -1) @ class_color_matrix  # [p(r|p), p(y|p), p(g|p)]
            color_likelihoods = color_likelihoods.reshape(-1, )

            # TODO -- optimize
            for ic, c in enumerate(["red", "yellow", "green"]):
                # P(c | p_k)
                p_col = color_likelihoods[ic]

                if p_col > 0:
                    # Posterior class probabilities if we observe color
                    cls_likelihood = class_color_matrix[:, ic]
                    gamma = prior * cls_likelihood
                    if np.sum(gamma) > 1e-6:
                        gamma = gamma / np.sum(gamma)
                        # Expected log Bayes factor
                        utility += p_col * np.sum(gamma * np.log(cls_likelihood) - np.log(p_col))

            # Add to list
            indicator_tuples.append((can, utility))

            # Update the utility of this indicator
            obs = self.indicators.observations.sel(indicator=can['name']).item()
            if obs == 0:
                # First observation
                self.indicators.observations.loc[can['name']] = 1
                self.indicators.utility.loc[can['name']] = utility
            else:
                # Running average
                current_avg = self.indicators.utility.loc[can['name']].item()
                self.indicators.observations.loc[can['name']] = obs + 1
                self.indicators.utility.loc[can['name']] += ((utility - current_avg) / obs)
                # self.indicator_utility_avg[j, 1] += ((utility - self.indicator_utility_avg[j, 1]) /
                #                                      self.indicator_utility_avg[j, 0])

        # Return the sorted list of candidates, descending
        return [tup[0] for tup in sorted(indicator_tuples, key=lambda x: x[1], reverse=True)]

    def process_asset(
        self, asset: Union[xr.Dataset, str], max_budget: float = None, pre_extracted: dict | None = None
    ) -> tuple[bool, Optional[str], float]:
        """
        Process an asset to infer its class and growth forecast

        :param asset:               xr.Dataset for the selected asset
        :param max_budget:          Max budget to use for this asset
        :param pre_extracted:       Optional dictionary of extracted indicators colors for this asset,
                                    {<indicator>: <value>}

        :return:    Boolean indicating success, and an optional string reason on failure, and remaining budget
        """
        start = iter_start = time.time()
        budget = min(max_budget, self.budget) if max_budget is not None else self.budget

        if isinstance(asset, str):
            # Get the asset by its name
            asset = self.stocks.sel(asset=asset)

        asset_id = str(asset.asset.values)

        # Get most recent alphas
        alphas = self.stocks.alphas.sel(asset=asset.asset).values[:, -1].copy()
        p_k = self.stocks.prior.sel(asset=asset.asset).values[:, -1].copy()

        # Special ranking index sorts by quant vs. qual, then utility within.
        indicator_items = [
            {
                "idx": idx,
                "name": self.indicators.indicator[idx].item(),
                "qualitative": self.indicators.qualitative[idx].item(),
                "cost": self.indicators.cost[idx].item(),
                "utility": self.indicators.utility[idx].item(),
            }
            for idx in self.sp_ranking_idx
        ]
        candidates = copy.deepcopy(indicator_items)  # For modification and removal during iteration

        # Shorthand
        color_to_ind = {
            "red": 0, "yellow": 1, "green": 2
        }

        # Generate hard indicators from asset financials
        # Uses the synth.INDEPENDENT_INDICATOR_INTERVALS to map from float to color
        if not pre_extracted:
            quant_indicators = synth.indicators_from_financials(
                self.stocks_json[asset_id]['financials'], as_colors=True
            )
        else:
            quant_indicators = {
                ind: pre_extracted[ind] for ind in synth.Indicators.quantitative
            }

        qual_processed = False  # Set true on successful qual
        early_stop = False  # Flag early stopping
        with logging_redirect_tqdm(loggers=[logger]):
            # We will at most to one evaluation for each indicator
            for i_ind in tqdm(range(Indicators.count), desc="Indicator", leave=False):
                logger.debug(f"Remaining asset budget: {budget}")

                # Filter candidates based on remaining budget
                for c in candidates.copy():
                    if c['cost'] > budget:
                        logger.debug(
                            f"Skipping '{c['name']}' due to expected cost: {c['cost']}"
                        )
                        candidates.remove(c)

                if not candidates:
                    break

                ####################
                # Quantitative stage

                # We first see if the next-up indicator is quantitative
                # If it is, we can skip the ranking algorithm and immediately process
                ind = indicator_items[i_ind]

                if ind['qualitative'] is False:
                    # Quickly process the quant indicator
                    try:
                        candidates.remove(ind)  # pop
                    except ValueError:
                        # Removed from budget filter
                        continue
                    color = quant_indicators[ind['name']]
                    color_pred = np.eye(3)[color_to_ind[color]]
                    likelihood = self.ind_likelihood(ind['name']) @ color_pred.reshape(-1, 1)
                    likelihood = likelihood.reshape(3, )

                    # p_k, alphas = cristal_stats.dirichlet_bayes(likelihood, p_k, alphas)
                    codegen_input = {
                        "likelihood": likelihood,
                        "p_k": p_k,
                        "alphas": alphas,
                    }
                    p_k, alphas = self.codegen_pipe("dirichlet_bayes", codegen_input)

                    # Update
                    self._update_alphas(asset, alphas)
                    self._update_prior(asset, p_k)
                    self.compute_growth_quantiles(asset=asset)
                    # update beta (the likelihood matrix here)
                    # self._update_likelihoods(ind['name'], color_pred)
                    continue

                ###################
                # Qualitative stage

                # Rank the remaining indicator candidates that are qualitative and within budget
                next_ranking = self.rank_indicators(p_k, candidates)
                # This shouldn't occur, but...
                assert next_ranking, "Failed to rank indicators"

                # Select best
                selected = next_ranking[0]
                selected_name = selected['name']
                # Assert it isn't quantitative
                assert selected['qualitative'], (f"Quantitative indicator '{selected_name}' leaked into Qualitative "
                                                 "evaluation stage")

                candidates.remove(selected)
                logger.debug(
                    f"Selecting '{selected_name}' | Expected cost {selected['cost']} | Remaining budget {budget})"
                )
                extraction_failed = False

                if pre_extracted:
                    # Convert from point estimate to distribution, like the synthetics
                    extracted_color_pred = np.eye(3)[color_to_ind[pre_extracted[selected_name]]]  # one-hot
                    se_cost = synth.synthetic_indicator_cost(selected['cost'])
                else:
                    # Call soft extractor
                    # Get report for this indicator, returns cached if available
                    reports_text = self.get_report(asset_id)
                    try:
                        extr_color, se_cost = self._call_soft_extractor(
                            asset, selected_name, budget, reports_text
                        )
                    except SoftIndicatorExtractionError as e:
                        logger.warning(f"Failed to extract indicator '{selected_name}'")
                        logger.debug(f"Failure Reason: {e}")
                        extraction_failed = True
                    else:
                        # Convert to one-hot distribution
                        extracted_color_pred = np.eye(3)[color_to_ind[extr_color]]
                        # if fails we should take the next indicator
                        # Update this indicators soft-extraction cost
                        # TODO -- Could be improved with an expected cost (same approach as for 'utility')
                        self.indicators.cost.loc[selected_name] = se_cost

                if not extraction_failed:
                    likelihood = self.ind_likelihood(selected_name) @ extracted_color_pred.reshape(-1, 1)
                    likelihood = likelihood.reshape(3, )
                    codegen_input = {
                        "likelihood": likelihood,
                        "p_k": p_k,
                        "alphas": alphas,
                    }
                    p_k, alphas = self.codegen_pipe("dirichlet_bayes", codegen_input)
                    # Update
                    qual_processed = True
                    self._update_alphas(asset, alphas)
                    # TODO -- Maybe move this out for performance
                    self.compute_growth_quantiles(asset=asset)
                    self._update_prior(asset, p_k)
                    # Early stopping if prior for one class > 0.95 or plateaus
                    if np.max(p_k) > 0.95:
                        logger.debug(f"Asset prior for one class > 0.95. Early stopping")
                        early_stop = True

                    # Check for plateau
                    last_n_priors = self.stocks['prior'].loc[asset.asset][:, -5].values
                    # Check the last_n isn't from the initialized values
                    init_prior = np.ones(self.n_classes, ) / self.n_classes
                    if not np.allclose(last_n_priors, init_prior):
                        avg_difference = np.mean(abs(p_k - last_n_priors))
                        if avg_difference < 0.1:
                            logger.debug(f"Asset prior has plateaued. Early stopping")
                            early_stop = True

                ###############
                # Update Budget

                if self.budget_mode == BudgetMode.WALL:
                    cost = time.time() - iter_start
                    iter_start = time.time()
                    self.budget -= cost
                    budget -= cost
                elif self.budget_mode == BudgetMode.SYNTH:
                    self.budget -= se_cost
                    budget -= se_cost
                else:
                    raise NotImplementedError(f"Budget mode {self.budget_mode} not implemented")

                if early_stop:
                    budget = 0
                    break

                if budget <= 0:
                    logger.debug(f"Asset budget exhausted")
                    break

        if not qual_processed:
            logger.warning("No qualitative indicators were within budget.")
            # Zero out the budget to prevent infinite recursion
            budget = 0

        if budget > 0:
            logger.warning(f"Finished processing asset with budget remaining: {budget}")

        # Refresh indicator rank at the end
        # self.refresh_indicator_rank()

        return True, "", budget

    def _compute_growth_quantiles_codegen(self, asset: xr.Dataset):
        """ load the generated code from cristal.llm.codegen.compute_growth_quantiles_codegen and use it for the return statement """
        raise NotImplementedError("This method is not implemented in the current codegen version of CRISTAL")

    # maintain the whole structure of our process assets function for partial codegen so nothing breaks downstream
    def process_asset_partial_codegen(
        self, asset: Union[xr.Dataset, str], max_budget: float = None, pre_extracted: dict | None = None
    ) -> tuple[bool, Optional[str], float]:
        """
        Process an asset to infer its class and growth forecast

        :param asset:               xr.Dataset for the selected asset
        :param max_budget:          Max budget to use for this asset
        :param pre_extracted:       Optional dictionary of extracted indicators colors for this asset,
                                    {<indicator>: <value>}

        :return:    Boolean indicating success, and an optional string reason on failure, and remaining budget
        """
        start = iter_start = time.time()
        budget = min(max_budget, self.budget) if max_budget is not None else self.budget

        if isinstance(asset, str):
            # Get the asset by its name
            asset = self.stocks.sel(asset=asset)

        asset_id = str(asset.asset.values)

        # Get most recent alphas
        alphas = self.stocks.alphas.sel(asset=asset.asset).values[:, -1].copy()
        p_k = self.stocks.prior.sel(asset=asset.asset).values[:, -1].copy()

        # Special ranking index sorts by quant vs. qual, then utility within.
        indicator_items = [
            {
                "idx": idx,
                "name": self.indicators.indicator[idx].item(),
                "qualitative": self.indicators.qualitative[idx].item(),
                "cost": self.indicators.cost[idx].item(),
                "utility": self.indicators.utility[idx].item(),
            }
            for idx in self.sp_ranking_idx
        ]
        candidates = copy.deepcopy(indicator_items)  # For modification and removal during iteration

        # Shorthand
        color_to_ind = {
            "red": 0, "yellow": 1, "green": 2
        }

        # Generate hard indicators from asset financials
        # Uses the synth.INDEPENDENT_INDICATOR_INTERVALS to map from float to color
        if not pre_extracted:
            quant_indicators = synth.indicators_from_financials(
                self.stocks_json[asset_id]['financials'], as_colors=True
            )
        else:
            quant_indicators = {
                ind: pre_extracted[ind] for ind in synth.Indicators.quantitative
            }

        qual_processed = False  # Set true on successful qual
        early_stop = False  # Flag early stopping
        with logging_redirect_tqdm(loggers=[logger]):
            # We will at most to one evaluation for each indicator
            for i_ind in tqdm(range(Indicators.count), desc="Indicator", leave=False):
                logger.debug(f"Remaining asset budget: {budget}")

                # Filter candidates based on remaining budget
                for c in candidates.copy():
                    if c['cost'] > budget:
                        logger.debug(
                            f"Skipping '{c['name']}' due to expected cost: {c['cost']}"
                        )
                        candidates.remove(c)

                if not candidates:
                    break

                ####################
                # Quantitative stage

                # We first see if the next-up indicator is quantitative
                # If it is, we can skip the ranking algorithm and immediately process
                ind = indicator_items[i_ind]

                if ind['qualitative'] is False:
                    # Quickly process the quant indicator
                    try:
                        candidates.remove(ind)  # pop
                    except ValueError:
                        # Removed from budget filter
                        continue
                    color = quant_indicators[ind['name']]
                    color_pred = np.eye(3)[color_to_ind[color]]
                    likelihood = self.ind_likelihood(ind['name']) @ color_pred.reshape(-1, 1)
                    likelihood = likelihood.reshape(3, )
                    p_k, alphas = cristal_stats.dirichlet_bayes_codegen(likelihood, p_k, alphas)
                    # Update
                    self._update_alphas(asset, alphas)
                    self._update_prior(asset, p_k)
                    self._compute_growth_quantiles_codegen(asset=asset)
                    # update beta (the likelihood matrix here)
                    # self._update_likelihoods(ind['name'], color_pred)
                    continue

                ###################
                # Qualitative stage

                # Rank the remaining indicator candidates that are qualitative and within budget
                next_ranking = self.rank_indicators(p_k, candidates)
                # This shouldn't occur, but...
                assert next_ranking, "Failed to rank indicators"

                # Select best
                selected = next_ranking[0]
                selected_name = selected['name']
                # Assert it isn't quantitative
                assert selected['qualitative'], (f"Quantitative indicator '{selected_name}' leaked into Qualitative "
                                                 "evaluation stage")

                candidates.remove(selected)
                logger.debug(
                    f"Selecting '{selected_name}' | Expected cost {selected['cost']} | Remaining budget {budget})"
                )
                extraction_failed = False

                if pre_extracted:
                    # Convert from point estimate to distribution, like the synthetics
                    extracted_color_pred = np.eye(3)[color_to_ind[pre_extracted[selected_name]]]  # one-hot
                    se_cost = synth.synthetic_indicator_cost(selected['cost'])
                else:
                    # Call soft extractor
                    # Get report for this indicator, returns cached if available
                    reports_text = self.get_report(asset_id)
                    try:
                        extr_color, se_cost = self._call_soft_extractor(
                            asset, selected_name, budget, reports_text
                        )
                    except SoftIndicatorExtractionError as e:
                        logger.warning(f"Failed to extract indicator '{selected_name}'")
                        logger.debug(f"Failure Reason: {e}")
                        extraction_failed = True
                    else:
                        # Convert to one-hot distribution
                        extracted_color_pred = np.eye(3)[color_to_ind[extr_color]]
                        # if fails we should take the next indicator
                        # Update this indicators soft-extraction cost
                        # TODO -- Could be improved with an expected cost (same approach as for 'utility')
                        self.indicators.cost.loc[selected_name] = se_cost

                if not extraction_failed:
                    likelihood = self.ind_likelihood(selected_name) @ extracted_color_pred.reshape(-1, 1)
                    likelihood = likelihood.reshape(3, )
                    p_k, alphas = cristal_stats.dirichlet_bayes_codegen(likelihood, p_k, alphas)
                    # Update
                    qual_processed = True
                    self._update_alphas(asset, alphas)
                    # TODO -- Maybe move this out for performance
                    self._compute_growth_quantiles_codegen(asset=asset)
                    self._update_prior(asset, p_k)
                    # Early stopping if prior for one class > 0.95 or plateaus
                    if np.max(p_k) > 0.95:
                        logger.debug(f"Asset prior for one class > 0.95. Early stopping")
                        early_stop = True

                    # Check for plateau
                    last_n_priors = self.stocks['prior'].loc[asset.asset][:, -5].values
                    # Check the last_n isn't from the initialized values
                    init_prior = np.ones(self.n_classes, ) / self.n_classes
                    if not np.allclose(last_n_priors, init_prior):
                        avg_difference = np.mean(abs(p_k - last_n_priors))
                        if avg_difference < 0.1:
                            logger.debug(f"Asset prior has plateaued. Early stopping")
                            early_stop = True

                ###############
                # Update Budget

                if self.budget_mode == BudgetMode.WALL:
                    cost = time.time() - iter_start
                    iter_start = time.time()
                    self.budget -= cost
                    budget -= cost
                elif self.budget_mode == BudgetMode.SYNTH:
                    self.budget -= se_cost
                    budget -= se_cost
                else:
                    raise NotImplementedError(f"Budget mode {self.budget_mode} not implemented")

                if early_stop:
                    budget = 0
                    break

                if budget <= 0:
                    logger.debug(f"Asset budget exhausted")
                    break

        if not qual_processed:
            logger.warning("No qualitative indicators were within budget.")
            # Zero out the budget to prevent infinite recursion
            budget = 0

        if budget > 0:
            logger.warning(f"Finished processing asset with budget remaining: {budget}")

        # Refresh indicator rank at the end
        # self.refresh_indicator_rank()

        return True, "", budget

    def _call_soft_extractor(
        self, asset: xr.Dataset, indicator_name: str, budget: float, reports_text: str | None = None
    ) -> tuple:
        """
        Call the Soft-Indicator Extractor to get color prediction from asset company booklet

        :param asset:           Selected asset
        :param indicator_name:  Soft indicator name
        :param budget:          Budget for this extraction
        """
        se_start = time.time()
        response, success = self.llm.extract_indicator(
            indicator_name, str(asset.asset), reports_text=reports_text, budget=budget
        )
        if not success:
            raise SoftIndicatorExtractionError(
                f"Soft indicator extraction failed: {response['error']}\n\nLast Completion:\n"
                f"{response['last_completion']}"
            )

        result = response["value"]

        if self.budget_mode == BudgetMode.SYNTH:
            _cost = self.indicators.cost.loc[indicator_name].item()
            cost = synth.synthetic_indicator_cost(_cost)
        elif self.budget_mode == BudgetMode.WALL:
            cost = time.time() - se_start
        else:
            raise NotImplementedError(f"Budget mode {self.budget_mode} not implemented")

        return result, cost

    def _synthetic_soft_extractor(self, indicator_name: str, synthetic_color: str) -> tuple:
        """
        Emulate SE call, return fake budget cost and noisy color transformation

        :param indicator_name:  Soft indicator name
        :param synthetic_color: Synthetic observed color
        :return: Tuple of color prediction, and simulated cost
        """
        color_to_ind = {
            "red": 0, "yellow": 1, "green": 2
        }

        extracted_color_pred = np.eye(3)[color_to_ind[synthetic_color]]  # one-hot

        # Simulate SE noise
        # extracted_color_pred = synth.synthetic_extractor_noise(extracted_color_pred)

        # Simulate SE cost
        cost = synth.synthetic_indicator_cost(self.indicators.cost.loc[indicator_name].item())

        return extracted_color_pred, cost

    def compute_growth_quantiles(
        self,
        quantiles: tuple[float] = (0.025, 0.16, 0.5, 0.84, 0.975),
        asset: Union[xr.Dataset, str] | None = None,
    ) -> None:
        """
        Compute the growth quantiles for either a specific asset, or if None, all assets
        Returns None because the values will be updated onto the stocks dataset

        :param quantiles:   List of quantiles to compute
        :param asset:       Optional asset to compute for
        :return:            None
        """
        if asset is None:
            for asset in self.stocks.asset.values:
                self.compute_growth_quantiles(quantiles, self.stocks.sel(asset=asset))
        else:
            # Compute the quantiles
            # We use the known Mixture means and stds, and then the asset mixture weights
            prior = self.stocks.prior.sel(asset=asset.asset).values[:, -1]
            quantile_values = synth.compute_growth_quantiles(prior, list(quantiles))
            self._update_quantile_pred(asset, quantile_values)

    def save_model(
        self,
        model_name: str | None = None,
        save_dir: str | None = None,
    ) -> bool:
        """
        Save the model in the form of its various attribute states:
            stocks:             The main stocks xr.Dataset.
                                This also contains the 'quantile_pred', 'alphas', and 'prior' histories
            ind_likelihoods:    The indicator likelihood matrix
            indicator_costs:    The expected indicator costs object
            indicator_utility_avg:  The running indicator utility average
            ranked_indicators:  Current indicator rankings
            llm:                The code of the LLM used
            mu_g, sigma_g:      The truth mu and sigma generators
            quantiles:          Quantiles

        :param model_name:  String name, relative to save_dir
        :param save_dir:    Optional save directory override
        """
        if not model_name:
            # The date as dd-mm-yyyy
            date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H%M")
            model_name = os.path.join(date_str, f"cristal_{self.llm.code}")

        if not save_dir:
            save_dir = os.path.join(PROJECT_ROOT, "models", model_name)

        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving model to {save_dir}")
        # First save the json data
        json_path = os.path.join(save_dir, "config.json")
        with open(json_path, 'w') as f:
            json.dump(self.to_json(), f, indent=4)

        try:
            # Save xr dataset individually
            self.stocks.to_netcdf(
                os.path.join(save_dir, "stocks.nc")
            )
            self.indicators.to_netcdf(
                os.path.join(save_dir, "indicator_data.nc")
            )

            # Save as npz
            npz_path = os.path.join(save_dir, "cristal.npz")
            np.savez(
                npz_path,
                indicator_costs=self.indicators.cost.values.tolist(),
                indicator_utility_avg=self.indicators.utility.values.tolist(),
                llm_code=self.llm.code,
                llm_temp=self.llm.temperature,
                mu_g=self.mu_g,
                sigma_g=self.sigma_g,
                quantiles=self.quantiles,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    @classmethod
    def load_model(cls, model_dir: str) -> "CRISTAL":
        """
        Load model from its directory
        """
        logger.info(f"Loading model from {model_dir}...")
        # Check this is valid dir
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' not found")

        stocks_path = os.path.join(model_dir, "stocks.nc")
        ind_path = os.path.join(model_dir, "indicator_data.nc")
        model_path = os.path.join(model_dir, "cristal.npz")

        stocks = xr.open_dataset(stocks_path)
        indicator_data = xr.open_dataset(ind_path)
        model_data = np.load(model_path, allow_pickle=True)

        llm_code = model_data['llm_code'].item()
        llm_temp = model_data['llm_temp'].item()
        mu_g = model_data['mu_g']
        sigma_g = model_data['sigma_g']
        quantiles = model_data['quantiles']

        # We infer the memory depth from stocks dataset
        asset_memory_depth = stocks.time.size

        if llm_code not in AVAILABLE_MODELS:
            logger.warning(f"LLM code '{llm_code}' not found in available models. No LLM initialized.")
            llm = None
        else:
            llm = AVAILABLE_MODELS[llm_code](
                temperature=llm_temp
            )

        # Create the model
        m = cls(
            stocks, llm, asset_memory_depth=asset_memory_depth,
        )
        # Set the loaded data
        m.indicators = indicator_data
        m.mu_g = mu_g
        m.sigma_g = sigma_g
        m.quantiles = quantiles

        logger.info("Model loaded successfully")
        return m

    def plot_history(self, asset: str, save_path: str | None = None):
        """
        Plot various information about an asset's history
            - Axis 1: Asset's asset-class predictions
            - Axis 2: Asset's growth quantile predictions

        :param asset:       str identifying the asset
        :param save_path:   path to save the plot
        :return:
        """
        # TODO -- Should be separated out of CRISTAL to a plot.py file
        # Initialize seaborn
        sns.set_theme(style="darkgrid")
        sns.set_context("notebook")

        asset = self.stocks.sel(asset=asset)
        asset_class = str(asset.category.values)
        time = self.stocks.time.values
        # Time as history, for display
        history = -time[::-1]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Plot class probabilities
        alphas = self.stocks.alphas.sel(asset=asset.asset).values
        prior = self.stocks.prior.sel(asset=asset.asset).values

        alpha_sum = np.sum(alphas, axis=0)
        for name, k in self.class_idx.items():
            lbl = self.classes[name]
            if name == asset_class:
                # True class
                color = 'g'
                ls = '-'
                lbl += " (true)"
            else:
                color = 'gray'
                ls = '--'

            # Expectation and confidence intervals from Dirichlet parameters
            alpha_k = alphas[k, :]
            # pi_k = alpha_k / alpha_sum
            var = alpha_k * (alpha_sum - alpha_k) / (alpha_sum ** 2 * (alpha_sum + 1))
            std = np.sqrt(var)
            pi_k = prior[k, :]

            ax1.plot(history, pi_k, color=color, linestyle=ls, label=lbl)
            ax1.fill_between(
                history,
                np.clip(pi_k - 2 * std, 0, 1),
                np.clip(pi_k + 2 * std, 0, 1),
                color=color, alpha=0.2
            )

        ax1.set_ylabel('Class Probability')
        ax1.legend(fontsize='small')
        # ax1.grid(True)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot growth estimate history, including the truth value for its asset-class
        # The quantile values will have shape [5, time]
        growth_quantiles = self.stocks.quantile_pred.sel(asset=asset.asset).values
        # Colors -- Get 5 colors from viridis
        colors = sns.color_palette("viridis", 5)
        q_labels = [
            "q-2.5",
            "q-16",
            "q-50",
            "q-84",
            "q-97.5",
        ]
        for i, q in enumerate([0.025, 0.16, 0.5, 0.84, 0.975]):
            ax2.plot(history, growth_quantiles[i, :], color=colors[i], label=q_labels[i])

        # Add horizontal like of the truth value
        asset_class = str(asset.category.values)
        asset_idx = self.class_idx[asset_class]
        mu_g = self.mu_g[asset_idx]
        ax2.axhline(mu_g, color='g', linestyle='--', label='True Growth')
        # Add known std
        sigma_g = self.sigma_g[asset_idx]
        ax2.fill_between(history, mu_g - sigma_g, mu_g + sigma_g, color='g', alpha=0.2)

        ax2.set_xlabel('History')
        ax2.set_ylabel('Predicted Growth')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Add legend and place outside the axis on the right
        ax2.legend(
            title="Quantiles", title_fontsize='medium',
            fontsize='small', ncol=1,
            bbox_to_anchor=(1.01, 1), loc='upper left',
        )
        # Repeat for axis 1
        ax1.legend(
            title="Asset Class", title_fontsize='medium',
            fontsize='small', ncol=1,
            bbox_to_anchor=(1.01, 1), loc='upper left',
        )
        # Shift the axes left for legend room
        fig.subplots_adjust(
            right=0.8,
        )

        if save_path:
            plt.savefig(save_path)
            print(f"Saving plot to {save_path}")

        plt.show()


def prompt_cristal_briefing() -> str:
    """
    Provide CRISTAL with a description of what it has access to and how it might use it:
    Make use of utils.describe_class.
    """
    prompt = """
    You will be prompted to write some code, e.g. statistical algorithms, to achieve various tasks.
    These snippets will be used by the CRISTAL class as part of its asset evaluation.
    Thus, you should make references to `self` when appropriate (you are the CRISTAL class).
    Below is a description of the CRISTAL class and its methods:\n\n
    """
    prompt += describe_class(CRISTAL)

    return prompt
