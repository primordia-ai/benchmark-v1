"""
ALLA Models
"""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from tqdm import tqdm
import random

from alla.utils import stats as alla_stats, synthetic as synth


class ALLA:
    """
    Active Lifelong Learning Architecture

    For our benchmarking experiment purposes.
    Stores a belief model of stock classes and their indicator likelihoods, etc.
    Stocks are one of High Growth, Stable, or Time Bomb.
    """

    def __init__(self, stocks_dataset: xr.Dataset, asset_memory_depth: int = 20):
        """
        Copy the stocks Dataset, and add new dimensions for ALLA's class prediction, and quantile predictions

        :param stocks_dataset:      xr.Dataset from synth.generate_synthetic_stocks
        :param asset_memory_depth:  How many Bayesian updates to remember for each asset
        """
        # Copy the stocks Dataset and add new dimension for ALLA's class prediction, and quantile predictions
        self.stocks: xr.Dataset = stocks_dataset.copy()
        self.asset_memory_depth = asset_memory_depth

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

        # Initialize belief model
        self.soft_indicators = stocks_dataset.coords['indicator'].values[
            stocks_dataset.coords['indicator_type'].values == 'soft'
        ]
        self.n_soft = len(self.soft_indicators)

        # For soft indicator likelihoods, we marginalize over colors
        marginals = synth.get_soft_likelihoods()
        self.soft_likelihoods = np.full((len(self.soft_indicators), self.n_classes), np.nan)  # likelihood array
        for key, val in marginals.items():
            self.soft_likelihoods[:, self.class_idx[key]] = val

        # Ensure no nans left
        assert not np.any(np.isnan(self.soft_likelihoods))

        self.soft_indicator_costs: dict = synth.SOFT_INDICATOR_COSTS

        # Init our results data, essentially
        self.indicator_utility_avg = None
        self.reset()

    def reset(self):
        """
        Reset the model to its initial state
        """
        # Quantile predictions -- All values initialized to 0
        # Updates will push oldest values out, where idx 0 is oldest
        # TODO -- update these to actual quantiles
        self.stocks['quantile_pred'] = xr.DataArray(
            np.zeros((self.stocks.sizes['asset'], 3, self.asset_memory_depth)),
            dims=['asset', 'quantile', 'time'],
            coords={
                # 'quantile': [0.025, 0.16, 0.5, 0.84, 0.975],
                'quantile': [0.33, 0.5, 0.66],
                'time': list(range(self.asset_memory_depth))
            }
        )
        # Asset priors -- All values initialized to 1/3 (across time as well)
        # Updates will push oldest values out, where idx 0 is oldest
        self.stocks['asset_prior'] = xr.DataArray(
            np.full((self.n_assets, self.n_classes, self.asset_memory_depth), 1 / self.n_classes),
            dims=['asset', 'asset_class', 'time'],
            coords={
                'asset_class': list(self.classes.keys()),
                'time': list(range(self.asset_memory_depth))
            },
        )

        # Indicator utility -- how much information in each indicator (for ranking)
        # We use a running average where [i, 0] and [i, 1] are the number of observations and current mean
        # for indicator i, respectively
        self.indicator_utility_avg = np.zeros((self.n_soft, 2))
        self.indicator_utility_avg[:, 1] = -np.inf

    def _update_quantile_pred(self, asset: xr.Dataset, quantile_pred):
        """
        Update the quantile predictions for an asset

        :param asset: xr.Dataset for the selected asset
        """
        # Update the quantile predictions -- place the new value at the end
        self.stocks['quantile_pred'].loc[asset.asset] = np.roll(
            self.stocks['quantile_pred'].loc[asset.asset], -1, axis=1
        )
        self.stocks['quantile_pred'].loc[asset.asset][:, -1] = quantile_pred

    def _update_asset_prior(self, asset: xr.Dataset, asset_prior):
        """
        Update the asset priors for an asset

        :param asset: xr.Dataset for the selected asset
        """
        # Update the asset priors -- place the new value at the end
        self.stocks['asset_prior'].loc[asset.asset] = np.roll(
            self.stocks['asset_prior'].loc[asset.asset], -1, axis=1
        )
        self.stocks['asset_prior'].loc[asset.asset][:, -1] = asset_prior

    def select_next_indicator(self, prior, candidates):
        """
        Select next indicator using expected log Bayes factor for growth prediction

        Also updates observed indicator utility average
        """
        max_utility = -float('inf')
        best_j = None

        # Current probabilities
        p_k = prior / np.sum(prior)

        # Current growth mixture params
        # prior_mean, prior_std = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)

        for j in candidates:
            utility = 0

            for x_ij in [0, 1]:
                # P(x_ij | current beliefs)
                p_xj = np.sum(
                    p_k * (self.soft_likelihoods[j, :] ** x_ij * (1 - self.soft_likelihoods[j, :]) ** (1 - x_ij))
                )

                if p_xj > 0:  # only compute if observation is possible
                    # Posterior class probabilities if we observe x_ij
                    likelihood = self.soft_likelihoods[j, :] ** x_ij * (1 - self.soft_likelihoods[j, :]) ** (1 - x_ij)
                    gamma = p_k * likelihood
                    gamma = gamma / np.sum(gamma)

                    # Posterior growth mixture params
                    # post_mean, post_std = alla_stats.mixture_stats(gamma, self.mu_g, self.sigma_g)

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
                            p_k[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                for k in range(self.n_classes)
                        )

                        if post_prob > 0 and prior_prob > 0:
                            # Weight by posterior probability of g
                            utility += p_xj * post_prob * (np.log(post_prob) - np.log(prior_prob)) * \
                                       (g_points[1] - g_points[0])

            # Update the utility of this indicator
            if self.indicator_utility_avg[j, 0] == 0:
                # First observation
                self.indicator_utility_avg[j, 0] = 1
                self.indicator_utility_avg[j, 1] = utility
                avg_utility = utility
            else:
                # Running average
                self.indicator_utility_avg[j, 0] += 1
                self.indicator_utility_avg[j, 1] += ((utility - self.indicator_utility_avg[j, 1]) /
                                                     self.indicator_utility_avg[j, 0])
                avg_utility = self.indicator_utility_avg[j, 1]

            if avg_utility > max_utility:
                max_utility = avg_utility
                best_j = j

        return best_j, max_utility

    def select_next_indicator_with_budget(self, prior, candidates, budget: int) -> tuple:
        """
        Select the next indicator to observe given a budget constraint.
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
        Returns:
        --------
        tuple
            A tuple containing the selected indicator and its associated utility.
        """
        
        max_utility = -float('inf')
        best_j = None

        # Current probabilities
        p_k = prior / np.sum(prior)

        # Current growth mixture params
        # prior_mean, prior_std = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)
        for j in candidates:
            utility = 0
            # check if we can afford this indicator
            if self.soft_indicator_costs[j] > budget:
                continue

            for x_ij in [0, 1]:
                # P(x_ij | current beliefs)
                p_xj = np.sum(
                    p_k * (self.soft_likelihoods[j, :] ** x_ij * (1 - self.soft_likelihoods[j, :]) ** (1 - x_ij))
                )

                if p_xj > 0:  # only compute if observation is possible
                    # Posterior class probabilities if we observe x_ij
                    likelihood = self.soft_likelihoods[j, :] ** x_ij * (1 - self.soft_likelihoods[j, :]) ** (1 - x_ij)
                    gamma = p_k * likelihood
                    gamma = gamma / np.sum(gamma)

                    # Posterior growth mixture params
                    # post_mean, post_std = alla_stats.mixture_stats(gamma, self.mu_g, self.sigma_g)

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
                            p_k[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                for k in range(self.n_classes)
                        )

                        if post_prob > 0 and prior_prob > 0:
                            # Weight by posterior probability of g
                            utility += p_xj * post_prob * (np.log(post_prob) - np.log(prior_prob)) * \
                                       (g_points[1] - g_points[0])

            # Update the utility of this indicator
            if self.indicator_utility_avg[j, 0] == 0:
                # First observation
                self.indicator_utility_avg[j, 0] = 1
                self.indicator_utility_avg[j, 1] = utility
                avg_utility = utility
            else:
                # Running average
                self.indicator_utility_avg[j, 0] += 1
                self.indicator_utility_avg[j, 1] += ((utility - self.indicator_utility_avg[j, 1]) /
                                                     self.indicator_utility_avg[j, 0])
                avg_utility = self.indicator_utility_avg[j, 1]

            if avg_utility > max_utility:
                max_utility = avg_utility
                best_j = j
            #given the calculated expected utility, check if we should use the indicator
            
            #update the budget
            budget -= self.soft_indicator_costs[j]
            if budget <= 0:
                break
        return best_j, max_utility

    def select_next_indicator_with_budget_evolutionary(self, prior, candidates, budget: int, population_size=2, generations=2, mutation_rate=0.1) -> tuple:
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
                    p_k * (self.soft_likelihoods[indicator, :] ** x_ij * (1 - self.soft_likelihoods[indicator, :]) ** (1 - x_ij))
                )
                if p_xj > 0:
                    likelihood = self.soft_likelihoods[indicator, :] ** x_ij * (1 - self.soft_likelihoods[indicator, :]) ** (1 - x_ij)
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
                            utility += p_xj * post_prob * (np.log(post_prob) - np.log(prior_prob)) * (g_points[1] - g_points[0])
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
                if self.soft_indicator_costs[ind] <= budget:
                    ind_utility = fitness(ind)
                    if ind_utility > max_utility:
                        max_utility = ind_utility
                        best_indicator = ind
        print(f"Best indicator: {best_indicator} with utility: {max_utility}")
        return best_indicator, max_utility

    def process_asset(self, asset: Union[xr.Dataset, str], soft_only=True) -> dict:
        """
        Process an asset to infer its class and growth forecast
        """
        if not soft_only:
            raise NotImplementedError("Only soft indicators are supported for now.")

        if isinstance(asset, str):
            # Get the asset by its name
            asset = self.stocks.sel(asset=asset)

        # Get the soft indicators, by masking with indicator_type
        indicators = asset.val.values[asset.indicator_type.values == 'soft']
        # Get most recent prior
        prior = self.stocks.asset_prior.sel(asset=asset.asset).values[:, -1]
        candidates = list(range(self.n_soft))

        selected_js = []
        utilities = []
        class_probs = []
        prior_history = []
        growth_means = []
        growth_stds = []

        for count in tqdm(range(len(indicators)), desc="Soft Ind."):
            prior_history.append(prior.copy())
            p_k = prior / np.sum(prior)
            class_probs.append(p_k)

            # Track growth predictions
            mean_g, std_g = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)
            growth_means.append(mean_g)
            growth_stds.append(std_g)

            j, utility = self.select_next_indicator(prior, candidates)
            utilities.append(utility)
            selected_js.append(j)
            candidates.remove(j)

            # Observed
            observed = indicators[j]

            likelihood = self.soft_likelihoods[j, :] ** observed * (1 - self.soft_likelihoods[j, :]) ** (1 - observed)
            posterior = p_k * likelihood
            posterior = posterior / np.sum(posterior)
            prior = posterior * np.sum(prior)

            # Update the stocks dataset with class probabilities and growth quantiles
            self._update_asset_prior(asset, prior)
            # TODO -- Calculate the quantiles properly
            quantiles = [
                mean_g - std_g,
                mean_g,
                mean_g + std_g
            ]
            self._update_quantile_pred(asset, quantiles)

        # return {
        #     "selected_js": selected_js,
        #     "utilities": utilities,
        #     "class_probs": class_probs,
        #     "prior_history": prior_history,
        #     "growth_means": growth_means,
        #     "growth_stds": growth_stds,
        # }

    def plot_prior_history(self, asset: str, save_path: str | None = None):
        """
        Plot the history of an asset's prior class probabilities and growth forecast
        TODO -- Should be separated out of ALLA to a plot.py file

        :param asset:       str identifying the asset
        :param save_path:   path to save the plot
        :return:
        """
        asset = self.stocks.sel(asset=asset)
        asset_class = str(asset.category.values)
        time = self.stocks.time.values

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot class probabilities
        class_probs = self.stocks.asset_prior.sel(asset=asset.asset).values
        for name, idx in self.class_idx.items():
            lbl = self.classes[name]
            if name == asset_class:
                # True class
                color = 'r'
                ls = '-'
                lbl += " (true)"
                true_mu_g = self.mu_g[idx]
            else:
                color = 'gray'
                ls = '--'

            ax2.plot(time, class_probs[idx, :], color=color, linestyle=ls, label=lbl)

            # ax2.fill_between(steps,
            #                  np.clip(class_probs[:, k] - 2 * std, 0, 1),
            #                  np.clip(class_probs[:, k] + 2 * std, 0, 1),
            #                  color=color, alpha=0.2)

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Class Probability')
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # # Plot growth forecast
        # growth_means = np.array(a0_results['growth_means'])
        # growth_stds = np.array(a0_results['growth_stds'])
        #
        # # Plot utilities
        # ax1.plot(steps, a0_results['utilities'], 'b-', label='Expected Log Bayes Factor for Growth')
        # ax1.set_xlabel('Step')
        # ax1.set_ylabel('Expected Utility')
        # ax1.legend()
        # ax1.grid(True)
        # ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if save_path:
            plt.savefig(save_path)
            print(f"Saving plot to {save_path}")

        plt.show()

    def process_asset_with_budget(self, asset: Union[xr.Dataset, str], soft_only=True, budget: int = 20) -> None:
        """
        Process an asset to infer its class and growth forecast
        """
        if not soft_only:
            raise NotImplementedError("Only soft indicators are supported for now.")

        if isinstance(asset, str):
            # Get the asset by its name
            asset = self.stocks.sel(asset=asset)

        # Get the soft indicators, by masking with indicator_type
        indicators = asset.val.values[asset.indicator_type.values == 'soft']
        # Get most recent prior
        prior = self.stocks.asset_prior.sel(asset=asset.asset).values[:, -1]
        candidates = list(range(self.n_soft))

        selected_js = []
        utilities = []
        class_probs = []
        prior_history = []
        growth_means = []
        growth_stds = []
        
        np.random.seed(42)
        costs = np.random.randint(1, 10, len(candidates))
        # Update soft indicator costs
        self.soft_indicator_costs = dict(zip(candidates, costs))
        print(f"Soft indicator costs: {self.soft_indicator_costs}")
        for count in tqdm(range(len(indicators)), desc="Soft Ind."):
            prior_history.append(prior.copy())
            p_k = prior / np.sum(prior)
            class_probs.append(p_k)

            # Track growth predictions
            mean_g, std_g = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)
            growth_means.append(mean_g)
            growth_stds.append(std_g)

            j, utility = self.select_next_indicator_with_budget_evolutionary(prior, candidates, budget=budget)
            utilities.append(utility)
            selected_js.append(j)
            candidates.remove(j)

            # Observed
            observed = indicators[j]

            likelihood = self.soft_likelihoods[j, :] ** observed * (1 - self.soft_likelihoods[j, :]) ** (1 - observed)
            posterior = p_k * likelihood
            posterior = posterior / np.sum(posterior)
            prior = posterior * np.sum(prior)

            # Update the stocks dataset with class probabilities and growth quantiles
            self._update_asset_prior(asset, prior)
            # TODO -- Calculate the quantiles properly
            quantiles = [
                mean_g - std_g,
                mean_g,
                mean_g + std_g
            ]
            self._update_quantile_pred(asset, quantiles)

        # return {
        #     "selected_js": selected_js,
        #     "utilities": utilities,
        #     "class_probs": class_probs,
        #     "prior_history": prior_history,
        #     "growth_means": growth_means,
        #     "growth_stds": growth_stds,
        # }