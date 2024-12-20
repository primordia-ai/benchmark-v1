"""
ALLA Models
"""
import logging
import os
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from pandas.conftest import observed
from scipy import stats
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from alla import stats as alla_stats, synthetic as synth
from alla.llm import Llama
from alla.utils import PROJECT_ROOT

logger = logging.getLogger("alla")


class ALLA:
    """
    Active Lifelong Learning Architecture

    For our benchmarking experiment purposes.
    Stores a belief model of stock classes and their indicator likelihoods, etc.
    Stocks are one of High Growth, Stable, or Time Bomb.
    """

    def __init__(
        self,
        stocks_dataset: xr.Dataset,
        asset_memory_depth: int = 20,
        budget: float = 100,
    ):
        """
        Copy the stocks Dataset, and add new dimensions for ALLA's class prediction, and quantile predictions

        :param stocks_dataset:      xr.Dataset from synth.generate_synthetic_stocks
        :param asset_memory_depth:  How many Bayesian updates to remember for each asset
        :param budget:              Budget for ALLA to spend on inference
        """
        # Copy the stocks Dataset and add new dimension for ALLA's class prediction, and quantile predictions
        self.stocks: xr.Dataset = stocks_dataset.copy()
        self.asset_memory_depth = asset_memory_depth
        self.budget = budget
        self.llm = Llama(temperature=0.)

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

        # List of soft indicator names
        self.soft_indicators = list(
            stocks_dataset.coords['indicator'].values[
                stocks_dataset.coords['indicator_type'].values == 'soft'
            ]
        )
        self.indicators = self.ordered_indicators = stocks_dataset.coords['indicator'].values.tolist()
        self.n_indicator = len(self.ordered_indicators)

        # Get the likelihood dataset. Dimensions are indicator, asset-class, and color
        self.ind_likelihoods: xr.Dataset = synth.get_indicator_dataset()

        # Integer costs by indicator name
        self.indicator_costs: dict = {}
        for ind in self.ordered_indicators:
            try:
                self.indicator_costs[ind] = synth.SOFT_INDICATOR_COSTS[ind]
            except KeyError:
                # Hard Indicator
                self.indicator_costs[ind] = 0.5

        # Init our results data, essentially
        self.indicator_utility_avg = None
        self.ranked_indicators = np.arange(self.n_indicator)
        self.reset()

    def reset(self):
        """
        Reset the model to its initial state
        """
        # Quantile predictions -- Quantiles initialized based on uniform class belief
        init_params = [(m, sig) for m, sig in zip(self.mu_g, self.sigma_g)]
        mix_weights = np.full(self.n_classes, 1 / self.n_classes)
        quantile_values = alla_stats.MixtureDistribution.compute_quantiles(
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

        # Indicator utility -- how much information in each indicator (for ranking)
        # We use a running average where [i, 0] and [i, 1] are the number of observations and current mean
        # for indicator i, respectively
        self.indicator_utility_avg = np.zeros((self.n_indicator, 2))
        self.indicator_utility_avg[:, 1] = -np.inf
        self.ranked_indicators = np.arange(self.n_indicator)

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

    def refresh_indicator_rank(self):
        """Returns the indicators ranked by utility average, descending"""
        self.ranked_indicators = np.argsort(self.indicator_utility_avg[:, 1])[::-1]

    def add_budget(self, budget: float):
        """
        Add budget to the model
        """
        self.budget += budget

    def rank_indicators(self, prior, candidates) -> list:
        """
        Rank indicators by expected utility using expected log Bayes factor for growth prediction

        Also updates observed indicator utility average
        """
        # For tracking utility
        indicator_tuples = []

        for j in candidates:
            utility = 0

            indicator_name = self.ordered_indicators[j]
            class_color_matrix = self.ind_likelihoods[indicator_name].values  # [classes, colors]
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
            indicator_tuples.append((j, utility))

            # Update the utility of this indicator
            if self.indicator_utility_avg[j, 0] == 0:
                # First observation
                self.indicator_utility_avg[j, 0] = 1
                self.indicator_utility_avg[j, 1] = utility
            else:
                # Running average
                self.indicator_utility_avg[j, 0] += 1
                self.indicator_utility_avg[j, 1] += ((utility - self.indicator_utility_avg[j, 1]) /
                                                     self.indicator_utility_avg[j, 0])

        # Return the sorted list of candidates, descending
        return [tup[0] for tup in sorted(indicator_tuples, key=lambda x: x[1], reverse=True)]

    def process_asset(self, asset: Union[xr.Dataset, str], max_budget: float = None) -> bool:
        """
        Process an asset to infer its class and growth forecast

        :param asset:       xr.Dataset for the selected asset
        :param max_budget:  Max budget to use for this asset
        :return:            Boolean indicating whether there was enough budget to perform at least one iteration
        """
        budget = min(max_budget, self.budget) if max_budget is not None else self.budget

        if isinstance(asset, str):
            # Get the asset by its name
            asset = self.stocks.sel(asset=asset)

        synthetic = False
        if synthetic:
            # Get the indicators from synthetic data
            observed_colors = asset.ryg.values
        else:
            # Get extracted color values
            # First look for save results
            extract_path = os.path.join(
                PROJECT_ROOT, "data/extracted_indicators", self.llm.code, asset.asset, ".json"
            )
            if os.path.exists(extract_path):
                logger.info(f"Loading extracted indicator results from '{extract_path}'")
                observed_colors =


        # Get most recent alphas
        alphas = self.stocks.alphas.sel(asset=asset.asset).values[:, -1].copy()
        p_k = self.stocks.prior.sel(asset=asset.asset).values[:, -1].copy()

        candidates = list(self.ranked_indicators.copy())
        selected_indicators = []
        growth_means = []
        growth_stds = []

        # Shorthand
        ind_cost = {
            j: self.indicator_costs[self.ordered_indicators[j]] for j in candidates
        }

        with logging_redirect_tqdm():
            for i_ind in tqdm(range(self.n_indicator), desc="Indicator"):
                logger.info(f"Remaining asset budget: {budget}")
                # Iterate through remaining candidates and filter by remaining budget
                for j in candidates.copy():
                    if ind_cost[j] > budget:
                        logger.debug(f"Skipping {j}: {self.ordered_indicators[j]} due to expected cost: {ind_cost[j]}")
                        candidates.remove(j)

                if not candidates:
                    break

                # Track growth predictions
                mean_g, std_g = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)
                growth_means.append(mean_g)
                growth_stds.append(std_g)

                next_ranking = self.rank_indicators(p_k, candidates)
                # This shouldn't occur, but...
                assert next_ranking, "Failed to rank indicators"

                # We should have already filtered out candidates with expected costs outside of budget
                # Therefore we pick the first ranked indicator
                selected_indicator = next_ranking[0]
                indicator_name = self.ordered_indicators[selected_indicator]
                selected_indicators.append(selected_indicator)
                candidates.remove(selected_indicator)
                logger.info(
                    f"Selecting {selected_indicator}: '{indicator_name}' with expected cost {ind_cost[selected_indicator]}"
                )

                # Call SE if we have a soft indicator
                if indicator_name in self.soft_indicators:
                    # extracted_color_pred, realized_cost = self._synthetic_soft_extractor(
                    #     asset, indicator_name, observed_colors[selected_indicator]
                    # )
                    extracted_color_pred, realized_cost = self._(
                        asset, indicator_name, observed_colors[selected_indicator]
                    )
                else:
                    color_to_ind = {
                        "red": 0, "yellow": 1, "green": 2
                    }
                    extracted_color_pred = np.eye(3)[color_to_ind[observed_colors[selected_indicator]]]  # one-hot
                    realized_cost = 0.5

                self.budget -= realized_cost
                budget -= realized_cost

                # Inference
                # Likelihood of asset-classes given color predictions
                A = self.ind_likelihoods[indicator_name].values  # [classes, colors]
                likelihood = A @ extracted_color_pred.reshape(-1, 1)
                likelihood = likelihood.reshape(3, )

                posterior = p_k * likelihood
                posterior /= np.sum(posterior)
                alphas += posterior
                p_k = posterior
                ###
                # MAP method.
                # From some brief experimenting, is essentially same outcome but very slightly worse variance
                # alphas_star = alphas + posterior - 1
                # alphas += alphas_star / np.sum(alphas_star)
                ###

                self._update_alphas(asset, alphas)
                self._update_prior(asset, posterior)
                self.compute_growth_quantiles(asset=asset)

                if budget <= 0:
                    logger.info(f"Budget exhausted after {i_ind} loops")
                    break

        # If no indicators were within budget, return False
        if len(selected_indicators) == 0:
            logger.warning("No indicators were within budget.")
            return False

        # Refresh indicator rank at the end
        self.refresh_indicator_rank()

        return True

    def _call_soft_extractor(
        self, asset: xr.Dataset, indicator_name: str) -> tuple:
        """
        Call the Soft-Indicator Extractor to get color prediction from asset company booklet

        :param asset:           Selected asset
        :param indicator_name:  Soft indicator name
        """
        return self.llm.extract_soft_indicator(asset, indicator_name)

    def _synthetic_soft_extractor(self, asset: xr.Dataset, indicator_name: str, observed_color: str) -> tuple:
        """
        Call Soft-Indicator Extractor to get color prediction
        Artificial at this point

        :param asset:           Selected asset
        :param indicator_name:  Soft indicator name
        :param observed_color:  Synthetic observed color
        :return: Tuple of color prediction, and simulated cost
        """
        color_to_ind = {
            "red": 0, "yellow": 1, "green": 2
        }

        extracted_color_pred = np.eye(3)[color_to_ind[observed_color]]  # one-hot
        # Simulate SE noise
        extracted_color_pred = synth.synthetic_extractor_noise(extracted_color_pred)
        # Simulate SE cost
        cost = synth.synthetic_indicator_cost(self.indicator_costs[indicator_name])

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
            generator_params = [(m, sig) for m, sig in zip(self.mu_g, self.sigma_g)]
            prior = self.stocks.prior.sel(asset=asset.asset).values[:, -1]
            # Normalize alphas to get mixture weights
            # weights = alphas / np.sum(alphas)

            quantile_values = alla_stats.MixtureDistribution.compute_quantiles(
                distributions=generator_params,
                weights=prior,
                quantiles=quantiles
            )
            self._update_quantile_pred(asset, quantile_values)

    def select_next_indicator_with_budget_evolutionary(
        self, prior, candidates, budget: int, population_size=2, generations=2, mutation_rate=0.1
    ) -> tuple:
        """
        Select the next indicator to observe given a budget constraint using an evolutionary algorithm.
        This method evaluates the utility of each candidate indicator and selects the one with the highest expected utility
        that can be afforded within the given budget. The utility is computed based on the expected log Bayes factor, which
        measures the information gain from observing the indicator.
        Parameters:
        -----------
        prior : np.ndarray
            The prior probabilities for each class.
        candidates : list
            A list of candidate indicators to consider.
        budget : int
            The available budget for selecting indicators.
        population_size : int
            The size of the population for the evolutionary algorithm.
        generations : int
            The number of generations to run the evolutionary algorithm.
        mutation_rate : float
            The mutation rate for the evolutionary algorithm.
        Returns:
        --------
        tuple
            A tuple containing the selected indicator and its associated utility.
        """

        def fitness(indicator):
            utility = 0
            p_k = prior / np.sum(prior)
            for x_ij in [0, 1]:
                p_xj = np.sum(
                    p_k * (self.ind_likelihoods[indicator, :] ** x_ij * (1 - self.ind_likelihoods[indicator, :]) ** (
                            1 - x_ij))
                )
                if p_xj > 0:
                    likelihood = self.ind_likelihoods[indicator, :] ** x_ij * (
                            1 - self.ind_likelihoods[indicator, :]) ** (1 - x_ij)
                    gamma = p_k * likelihood
                    gamma = gamma / np.sum(gamma)
                    g_points = np.linspace(
                        min(self.mu_g - 3 * max(self.sigma_g)),
                        max(self.mu_g + 3 * max(self.sigma_g)),
                        100
                    )
                    for g in g_points:
                        post_prob = sum(
                            gamma[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                for k in range(self.n_classes)
                        )
                        prior_prob = sum(
                            p_k[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                for k in range(self.n_classes)
                        )
                        if post_prob > 0 and prior_prob > 0:
                            utility += p_xj * post_prob * (np.log(post_prob) - np.log(prior_prob)) * (
                                    g_points[1] - g_points[0])
            return utility

        def mutate(indicator):
            if random.random() < mutation_rate:
                return random.choice(candidates)
            return indicator

        def crossover(parent1, parent2):
            return random.choice([parent1, parent2])

        # Initialize population
        population = [random.choice(candidates) for _ in range(population_size)]
        best_indicator = None
        max_utility = -float('inf')

        for generation in tqdm(range(generations)):
            # Evaluate fitness
            fitness_values = [fitness(ind) for ind in population]

            # Select parents
            parents = random.choices(population, weights=fitness_values, k=population_size)

            # Generate new population
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1 = mutate(crossover(parent1, parent2))
                child2 = mutate(crossover(parent1, parent2))
                new_population.extend([child1, child2])

            population = new_population

            # Update best indicator
            for ind in population:
                if self.indicator_costs[ind] <= budget:
                    ind_utility = fitness(ind)
                    if ind_utility > max_utility:
                        max_utility = ind_utility
                        best_indicator = ind
        print(f"Best indicator: {best_indicator} with utility: {max_utility}")
        return best_indicator, max_utility

    def plot_history(self, asset: str, save_path: str | None = None):
        """
        Plot various information about an asset's history
            - Axis 1: Asset's asset-class predictions
            - Axis 2: Asset's growth quantile predictions

        :param asset:       str identifying the asset
        :param save_path:   path to save the plot
        :return:
        """
        # TODO -- Should be separated out of ALLA to a plot.py file
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
        # Colors -- Get 5 colors from the icefire diverging palette
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
